using AlprNet.Lib;
using OpenCvSharp;

namespace AlprNet.ConsoleApp
{
    internal class Program
    {
        private static async Task Main(string[] args)
        {
            var image = Mat.FromImageData(await File.ReadAllBytesAsync("car_plate.jpg"));

            var modelDirPath = Path.Combine(Directory.GetCurrentDirectory(), "Models");
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
