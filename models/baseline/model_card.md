# Model Card: lane99/resnet_mnist_digits

## Model Details
- **Model name / ID:** lane99/resnet_mnist_digits
- **Provider:** Hugging Face
- **Revision / commit:** main
- **Architecture summary:** ResNet (Residual Network)
- **Intended task:** Image classification (digit recognition)
- **Input format:** 28x28 grayscale images (MNIST format)
- **Output format:** 10-class probability distribution (digits 0-9)

## Training Data
- **Dataset(s):** MNIST
- **Dataset version(s):** Standard MNIST
- **Preprocessing:** Standard normalization for MNIST

## Evaluation
- **Metrics used:** Accuracy, loss
- **Baseline performance (if known):** TBD
- **Known failure cases:** TBD

## Limitations & Risks
- **Operational constraints:** Designed for 28x28 grayscale images only
- **Bias / fairness considerations (as applicable):** Limited to printed/digital digits, may not generalize to handwritten digits in different contexts
- **Safety considerations:** None specific to digit classification

## License & Use
- **License:** See Hugging Face model page
- **Restrictions / attribution:** As specified by model author

## Provenance
- **Downloaded from:** https://huggingface.co/lane99/resnet_mnist_digits
- **Download date:** TBD
- **Stored in repo at:** models/baseline/
