FROM python:3.10-slim

# Install dependencies once so training runs quickly
RUN python -m pip install --no-cache-dir \
        datasets==2.18.0 \
        --extra-index-url https://pypi.org/simple \
    && python -m pip install --no-cache-dir 'numpy<2' \
    && python -m pip install --no-cache-dir \
        torch==2.2.1 \
        torchvision==0.17.1 \
        --index-url https://download.pytorch.org/whl/cpu

ENV OUTPUT_DIR=/data/outputs
WORKDIR /workspace
CMD ["python", "train_mnist.py", "--data-dir", "mnist-dataset/mnist", "--epochs", "1", "--batch-size", "256", "--output-dir", "/data/outputs"]
