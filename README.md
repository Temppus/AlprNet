# AlprNet

.NET standard library for license plate detection and recognition. It uses public and open source ONNX models for inference.

> **Note:** Library works, but it's WIP. Once more polished, a NuGet package will be published.

## Demo

### Input Image
<img src="AlprNet/AlprNet.ConsoleApp/car.jpg" alt="Car Image" style="max-width: 250px;">

### Detected License Plate
![Detected License Plate](AlprNet/AlprNet.ConsoleApp/car_plate.jpg)

**License plate string**: `UD 1234`

## Usage

Example usage. See [`Program.cs`](AlprNet.ConsoleApp/Program.cs) for full inference code.

```csharp
using AlprNet.Lib.PlateDetection;
using AlprNet.Lib.PlateRecognition;
using OpenCvSharp;

// Load an image
var image = Mat.FromImageData(await File.ReadAllBytesAsync("car.jpg"));

// Step 1: Detect license plates
var modelDirPath = Path.Combine(Directory.GetCurrentDirectory(), "PlateDetection\\Models");
var modelPath = Path.Combine(modelDirPath, "yolo-v9-t-512-license-plates-end2end.onnx");
using var licensePlateDetector = new LicensePlateDetector(modelPath);

var detections = licensePlateDetector.Run(image);
var detectedPlate = detections.Single();

// Step 2: Crop the detected plate
var plateCropImg = detectedPlate.ToPlateCropImage(image);

// Step 3: Recognize the plate text
var recognitionModelPath = Path.Combine(Directory.GetCurrentDirectory(), "PlateRecognition\\Models");
var onnxModelPath = Path.Combine(recognitionModelPath, "cct_s_v1_global.onnx");
var configPath = Path.Combine(recognitionModelPath, "cct_s_v1_global_plate_config.yaml");

using var recognizer = new LicensePlateRecognizer(onnxModelPath, configPath);
var result = recognizer.Run([plateCropImg], returnConfidence: true);

var plateText = result.plates.Single();
Console.WriteLine($"License plate: {plateText}");
```

## Models
Models are packed in repository, but they can also be downloaded via utility classes:
- `PlateDetectionModelsRepository`
- `PlateOcrModelsRepository`

## Acknowledgments

- YOLO v9 models and inference code for license plate detection (https://github.com/ankandrew/open-image-models)
- OCR models and inference code for license plate recognition (https://github.com/ankandrew/fast-alpr)