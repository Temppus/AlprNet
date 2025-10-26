namespace AlprNet.ConsoleApp.PlateDetection
{
    public static class PlateDetectionModelsRepository
    {
        public enum PlateDetectorModelType
        {
            YoloV9S608LicensePlateEnd2End,
            YoloV9T640LicensePlateEnd2End,
            YoloV9T512LicensePlateEnd2End,
            YoloV9T416LicensePlateEnd2End,
            YoloV9T384LicensePlateEnd2End,
            YoloV9T256LicensePlateEnd2End
        }

        private const string BaseUrl = "https://github.com/ankandrew/open-image-models/releases/download";

        public static readonly Dictionary<PlateDetectorModelType, string> AvailableOnnxModels = new()
        {
            [PlateDetectorModelType.YoloV9S608LicensePlateEnd2End] = $"{BaseUrl}/assets/yolo-v9-s-608-license-plates-end2end.onnx",
            [PlateDetectorModelType.YoloV9T640LicensePlateEnd2End] = $"{BaseUrl}/assets/yolo-v9-t-640-license-plates-end2end.onnx",
            [PlateDetectorModelType.YoloV9T512LicensePlateEnd2End] = $"{BaseUrl}/assets/yolo-v9-t-512-license-plates-end2end.onnx",
            [PlateDetectorModelType.YoloV9T416LicensePlateEnd2End] = $"{BaseUrl}/assets/yolo-v9-t-416-license-plates-end2end.onnx",
            [PlateDetectorModelType.YoloV9T384LicensePlateEnd2End] = $"{BaseUrl}/assets/yolo-v9-t-384-license-plates-end2end.onnx",
            [PlateDetectorModelType.YoloV9T256LicensePlateEnd2End] = $"{BaseUrl}/assets/yolo-v9-t-256-license-plates-end2end.onnx"
        };

        public static async Task DownloadAllModelsAsync(string targetDir)
        {
            foreach (var kvp in AvailableOnnxModels)
            {
                var url = kvp.Value;

                var fileName = Path.GetFileName(url);
                var destinationPath = Path.Combine(targetDir, fileName);

                if (File.Exists(destinationPath))
                    continue;

                await DownloadFileAsync(url, destinationPath);
            }
        }

        public static async Task<string> DownloadModelAsync(PlateDetectorModelType modelType, string targetDir, bool forceDownload = false)
        {
            if (!AvailableOnnxModels.TryGetValue(modelType, out var modelUrl))
                throw new ArgumentException($"Unknown model type: {modelType}");

            Directory.CreateDirectory(targetDir);

            var fileName = Path.GetFileName(modelUrl);
            var destinationPath = Path.Combine(targetDir, fileName);

            if (File.Exists(destinationPath) && !forceDownload)
            {
                Console.WriteLine($"Model '{modelType}' already exists at {destinationPath}. Skipping download.");
                return destinationPath;
            }

            Console.WriteLine($"Downloading model '{modelType}'...");
            await DownloadFileAsync(modelUrl, destinationPath);
            Console.WriteLine($"Model '{modelType}' downloaded to {destinationPath}");

            return destinationPath;
        }

        private static async Task DownloadFileAsync(string url, string destinationPath)
        {
            if (File.Exists(destinationPath))
                return;

            using var httpClient = new HttpClient();
            using var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            if (!response.IsSuccessStatusCode)
            {
                return;
            }

            await using var stream = await response.Content.ReadAsStreamAsync();
            await using var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None);
            await stream.CopyToAsync(fileStream);
        }
    }
}
