FROM public.ecr.aws/lambda/python:3.10.2024.10.30.16-x86_64

# Upgrade pip
RUN pip install --upgrade pip

# Copy project files to the container
COPY . .

RUN python -m pip install ".[cli]"

RUN pip install "numpy<2"

RUN pip install boto3

# Create the checkpoint directory
RUN mkdir -p rembg/checkpoint

# Download the model
RUN curl -L -o rembg/checkpoint/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx

# Define the Lambda handler
CMD ["main.handler"]
