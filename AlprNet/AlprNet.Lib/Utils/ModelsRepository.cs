using AlprNet.Lib.Config;

namespace AlprNet.Lib.Utils
{
    public static class ModelsRepository
    {
        private const string BaseUrl = "https://github.com/ankandrew/cnn-ocr-lp/releases/download";

        // Exact mapping from the Python dict
        private static readonly Dictionary<OcrModelType, (string OnnxUrl, string YamlUrl)> AvailableOnnxModels = new()
        {
            [OcrModelType.CctSV1GlobalModel] = (
                $"{BaseUrl}/arg-plates/cct_s_v1_global.onnx",
                $"{BaseUrl}/arg-plates/cct_s_v1_global_plate_config.yaml"
            ),
            [OcrModelType.CctXsV1GlobalModel] = (
                $"{BaseUrl}/arg-plates/cct_xs_v1_global.onnx",
                $"{BaseUrl}/arg-plates/cct_xs_v1_global_plate_config.yaml"
            ),
            [OcrModelType.ArgentinianPlatesCnnModel] = (
                $"{BaseUrl}/arg-plates/arg_cnn_ocr.onnx",
                $"{BaseUrl}/arg-plates/arg_cnn_ocr_config.yaml"
            ),
            [OcrModelType.ArgentinianPlatesCnnSynthModel] = (
                $"{BaseUrl}/arg-plates/arg_cnn_ocr_synth.onnx",
                $"{BaseUrl}/arg-plates/arg_cnn_ocr_config.yaml"
            ),
            [OcrModelType.EuropeanPlatesMobileVitV2Model] = (
                $"{BaseUrl}/arg-plates/european_mobile_vit_v2_ocr.onnx",
                $"{BaseUrl}/arg-plates/european_mobile_vit_v2_ocr_config.yaml"
            ),
            [OcrModelType.GlobalPlatesMobileVitV2Model] = (
                $"{BaseUrl}/arg-plates/global_mobile_vit_v2_ocr.onnx",
                $"{BaseUrl}/arg-plates/global_mobile_vit_v2_ocr_config.yaml"
            ),
            [OcrModelType.CctSReluV1GlobalModel] = (
                $"{BaseUrl}/arg-plates/cct_s_relu_v1_global.onnx",
                $"{BaseUrl}/arg-plates/cct_s_relu_v1_global_plate_config.yaml"
            ),
            [OcrModelType.CctXsReluV1GlobalModel] = (
                $"{BaseUrl}/arg-plates/cct_xs_relu_v1_global.onnx",
                $"{BaseUrl}/arg-plates/cct_xs_relu_v1_global_plate_config.yaml"
            )
        };

        public static async Task DownloadModelsAsync(string targetDir)
        {
            Console.WriteLine($"Downloading models to: {targetDir}\n");

            foreach (var kvp in AvailableOnnxModels)
            {
                var model = kvp.Key;
                var (onnxUrl, yamlUrl) = kvp.Value;

                await DownloadFileAsync(onnxUrl, Path.Combine(targetDir, Path.GetFileName(onnxUrl)));
                await DownloadFileAsync(yamlUrl, Path.Combine(targetDir, Path.GetFileName(yamlUrl)));
            }

            Console.WriteLine("\nAll downloads finished.");
        }

        private static async Task DownloadFileAsync(string url, string destinationPath)
        {
            // Skip if already present (optional – comment out to always re-download)
            if (File.Exists(destinationPath))
            {
                Console.WriteLine($"[SKIP] {Path.GetFileName(destinationPath)} already exists.");
                return;
            }

            Console.Write($"Downloading {Path.GetFileName(destinationPath)} ... ");

            try
            {
                var httpClient = new HttpClient();
                using var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                if (!response.IsSuccessStatusCode)
                {
                    Console.WriteLine($"FAILED ({response.StatusCode})");
                    return;
                }

                await using var stream = await response.Content.ReadAsStreamAsync();
                await using var fileStream = new FileStream(destinationPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                await stream.CopyToAsync(fileStream);

                Console.WriteLine("DONE");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: {ex.Message}");
            }
        }
    }
}
