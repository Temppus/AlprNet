using AlprNet.Lib.PlateDetection;
using AlprNet.Lib.PlateRecognition;
using OpenCvSharp;

namespace AlprNet.ConsoleApp
{
    internal class Program
    {
        private static async Task Main(string[] args)
        {
            Mat plateCropImg;

            // Plate detection
            {
                var modelDirPath = Path.Combine(Directory.GetCurrentDirectory(), "PlateDetection\\Models");
                var modelPath = Path.Combine(modelDirPath, "yolo-v9-t-512-license-plates-end2end.onnx");
                using var licensePlateDetector = new LicensePlateDetector(modelPath);
                var image = Mat.FromImageData(await File.ReadAllBytesAsync("car.jpg"));

                Console.WriteLine("Running license plate detection");
                var detections = licensePlateDetector.Run(image);
                var detectedPlate = detections.Single();

                plateCropImg = detectedPlate.ToPlateCropImage(image);
                Console.WriteLine("License plate detected and cropped");
                plateCropImg.SaveImage("car_plate.jpg");
            }

            // Plate recognition
            {
                var modelDirPath = Path.Combine(Directory.GetCurrentDirectory(), "PlateRecognition\\Models");
                var modelPath = Path.Combine(modelDirPath, "cct_s_v1_global.onnx");
                var modelConfigPath = Path.Combine(modelDirPath, "cct_s_v1_global_plate_config.yaml");
                using var recognizer = new LicensePlateRecognizer(modelPath, modelConfigPath);

                Console.WriteLine("Running license plate recognition");
                var result = recognizer.Run([plateCropImg], returnConfidence: true);

                var plate = result.plates.Single();
                Console.WriteLine($"License plate is: {plate}");
            }
        }

        private static async Task RunPlateRecognitionSampleAsync()
        {
            var image = Mat.FromImageData(await File.ReadAllBytesAsync("car_plate.jpg"));

            var modelDirPath = Path.Combine(Directory.GetCurrentDirectory(), "PlateRecognition\\Models");
            var files = Directory.GetFiles(modelDirPath);

            var onnxModelPaths = files.Where(x => x.EndsWith(".onnx")).ToArray();
            var configFilePaths = files.Where(x => x.EndsWith(".yaml")).ToArray();

            foreach (var modelPath in onnxModelPaths)
            {
                var modelFileName = Path.GetFileNameWithoutExtension(modelPath);

                try
                {
                    var modelConfigPath = configFilePaths.Single(x => new FileInfo(x).Name.StartsWith(modelFileName));

                    Console.WriteLine($"Running model: {modelFileName}");
                    using var recognizer = new LicensePlateRecognizer(modelPath, modelConfigPath);
                    var result = recognizer.Run([image], returnConfidence: true);

                    var plate = result.plates.Single();
                    var confidences = result.confidences;

                    Console.WriteLine($"License plate is: {plate}");

                    if (confidences != null)
                    {
                        Console.Write("Confidences: ");
                        int rows = confidences.GetLength(0);
                        int cols = confidences.GetLength(1);

                        for (int i = 0; i < rows; i++)
                        {
                            for (int j = 0; j < cols; j++)
                            {
                                Console.Write($"{confidences[i, j],8:F3}"); // Format: 8 chars wide, 4 decimal places
                            }

                            Console.WriteLine();
                        }
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine($"[ERROR] Fail to run inference for model {modelFileName}. Reason: {e.Message}");
                }
                finally
                {
                    Console.WriteLine("-----------------------------------------\n");
                }
            }
        }
    }
}
