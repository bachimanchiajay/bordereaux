from common import CallableComponent, ExtractionState
import json
from pydantic import ValidationError
from common import DirtyJsonParser
from src.visual_extract.helper.PromptBuilder import PromptBuilder

# Import your generation schemas
from extraction_io.generation_utils.TableGeneration import TableGeneration
from extraction_io.generation_utils.KeyValueGeneration import KeyValueGeneration
from extraction_io.generation_utils.SummaryGeneration import SummaryGeneration


class VLMProcessor(CallableComponent):
    """
    Handles extraction tasks using a Vision-Language Model (VLM) inference engine.
    """

    def __init__(self, vlm_infer, lm_processor=None):
        super().__init__()
        self.vlm_infer = vlm_infer
        self.lm_processor = lm_processor

    def extract(self, image_data, prompt, generation_model, **kwargs):
        """
        Runs the VLM inference on image_data with the given prompt,
        then parses the JSON and validates.
        """
        item = kwargs.get('item') if kwargs.get('item') else ExtractionState.get_current_extraction_item()

        parsed = None
        last_raw_output = None
        last_validation_error = None
        had_parse_error = False
        had_validation_error = False

        for i in range(2):
            self.logger.info("Running VLM inference on image_data...")
            raw_output = self.vlm_infer.infer(image_data, prompt)
            last_raw_output = raw_output
            self.logger.info("Finished VLM inference on image_data.")

            try:
                # Attempt to parse as JSON
                parsed = DirtyJsonParser.parse(raw_output)
            except json.JSONDecodeError as e:
                self.logger.warning(f"VLM output is not valid JSON (attempt {i+1}/2): {e}. Retrying...")
                had_parse_error = True
                retry = PromptBuilder.get_retry_prompt('parse_error')
                prompt = f"{prompt}\n\n{retry}" if retry else prompt
                continue

            # --- ✅ Add Fallback Handling for Summary ---
            if isinstance(parsed, str):
                # Sometimes summary output is plain text instead of JSON
                self.logger.info("Detected plain-text summary, wrapping into valid schema format...")
                parsed = {
                    "field_name": getattr(item, "field_name", "SummaryField"),
                    "value": parsed,
                    "continue_next_page": False
                }

            # --- ✅ Validate against generation model ---
            try:
                parsed = generation_model.model_validate(parsed)
                if item:
                    parsed.field_name = item.field_name
                return parsed

            except ValidationError as e:
                last_validation_error = e
                self.logger.warning(
                    f"VLM output failed schema validation (attempt {i+1}/2): {e}. Retrying..."
                )
                had_validation_error = True
                retry = PromptBuilder.get_retry_prompt('validation_error')
                prompt = f"{prompt}\n\n{retry}" if retry else prompt
                parsed = None
                continue

        # --- If we reach here, both attempts failed ---
        if had_parse_error and not had_validation_error:
            try:
                schema = generation_model.model_json_schema()
            except Exception:
                schema = {}

            try:
                schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
            except Exception:
                schema_json = str(schema)

            # Build LM repair prompt
            lm_prompt = PromptBuilder.build_lm_repair_prompt(
                schema_json=schema_json,
                previous_output=last_raw_output,
                last_error=str(last_validation_error)
            )

            try:
                self.logger.info("Attempting schema repair with LM (text-only)...")
                lm_processor = kwargs.get('lm_processor') or self.lm_processor
                if lm_processor is not None:
                    repaired = lm_processor(lm_prompt, generation_model, item=item)
                else:
                    lm_raw = self.vlm_infer.infer_lang(lm_prompt)
                    self.logger.debug(f"LM repair raw output (truncated): {str(lm_raw)[:500]}")
                    repaired = DirtyJsonParser.parse(lm_raw)
                    repaired = generation_model.model_validate(repaired)
                    if item:
                        repaired.field_name = item.field_name
                    self.logger.info("Schema repair succeeded via LM.")
                    return repaired
            except Exception as e:
                self.logger.error(f"Schema repair via LM failed: {e}. Last VLM validation error: {last_validation_error}")

        # Fallback return
        return parsed

    def __call__(self, image_data, prompt, generation_model, *args, **kwargs):
        self.logger.info(f"Executing extraction for model: {generation_model.__class__.__name__}")
        return self.extract(image_data, prompt, generation_model, **kwargs)
