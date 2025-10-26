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
                    var res = recognizer.Run([image], returnConfidence: false);
                    var plate = res.plates.Single();
                    Console.WriteLine($"License plate is: {plate}");
                }
                catch (Exception e)
                {
                    Console.WriteLine($"[ERROR] Fail to run inference for model {modelFileName}. Reason: {e.Message}");
                }
            }
        }
    }
}
