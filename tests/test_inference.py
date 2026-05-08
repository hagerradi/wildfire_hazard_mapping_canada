from inference.run_ai_surrogate_model_hexel_inference import get_target_spec_from_data_config


def test_inference_target_spec_defaults_to_bp_for_old_checkpoints():
    target = get_target_spec_from_data_config({"input_sources": [{"name": "grid", "params": {}}]})

    assert target.name == "bp"
    assert target.output_type == "fire_burn_probability"


def test_inference_target_spec_uses_checkpoint_target_name():
    target = get_target_spec_from_data_config({"input_sources": [{"name": "grid", "params": {"target_name": "ros"}}]})

    assert target.name == "ros"
    assert target.output_type == "fire_ros"
