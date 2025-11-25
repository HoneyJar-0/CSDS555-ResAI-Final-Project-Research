from pathlib import Path

from tensorboard import program
from configs import experiment_config


def launch_tensorboard():
    try:
        log_dir = Path(experiment_config.log_dir)
        if not log_dir.exists():
            print("Log directory does not exist.")
            return

        print(f"Launching TensorBoard for latest run folder: {log_dir}")
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", str(log_dir), "--port", str(experiment_config.tensorboard_port)])
        url = tb.launch()
        print(f"TensorBoard running at {url}")

    except Exception as e:
        print(f"Failed to launch TensorBoard: {e}")