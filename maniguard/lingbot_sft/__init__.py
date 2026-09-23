"""LingBot-VLA 2.0 configuration for ManiGuard joint-control datasets.

YAML files map state/actions to a single seven-joint arm plus one gripper
coordinate, and image_left/wrist_image to overview and wrist cameras. Actions
use subtract_state=false and are absolute joint targets. Unmapped dimensions
in the model's unified representation are padded or masked by its loader.

Training scripts run from a separately installed LingBot source tree. Copy
robot_config.yaml, train_config.yaml, and norm_compute_config.yaml to the
paths expected by tools/lingbot_sft/run_sft.sh. The driver supplies dataset,
normalization, output, and step settings per family."""
