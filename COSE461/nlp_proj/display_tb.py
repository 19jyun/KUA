import subprocess
import webbrowser
import time

#Last version before merging into multimodal

def launch_tensorboard():
    try:
        # TensorBoard를 실행합니다.
        subprocess.Popen(['tensorboard', '--logdir=logs'])
        time.sleep(2)  # Wait a bit for the server to start
        webbrowser.open('http://localhost:6006')
    except Exception as e:
        print(f"Failed to launch TensorBoard: {e}")

if __name__ == "__main__":
    launch_tensorboard()