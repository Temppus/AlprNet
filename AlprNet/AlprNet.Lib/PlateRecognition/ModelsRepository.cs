namespace AlprNet.Lib.PlateRecognition
{
    public static class ModelsRepository
    {
        public enum OcrModelType
        {
            CctSV1GlobalModel,
            CctXsV1GlobalModel,
            CctSReluV1GlobalModel,
            CctXsReluV1GlobalModel,
            ArgentinianPlatesCnnModel,
            ArgentinianPlatesCnnSynthModel,
            EuropeanPlatesMobileVitV2Model,
            GlobalPlatesMobileVitV2Model
        }

        private const string BaseUrl = "https://github.com/ankandrew/cnn-ocr-lp/releases/download";

        public static readonly Dictionary<OcrModelType, (string OnnxUrl, string YamlUrl)> AvailableOnnxModels = new()
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
            foreach (var kvp in AvailableOnnxModels)
            {
                var (onnxUrl, yamlUrl) = kvp.Value;

                await DownloadFileAsync(onnxUrl, Path.Combine(targetDir, Path.GetFileName(onnxUrl)));
                await DownloadFileAsync(yamlUrl, Path.Combine(targetDir, Path.GetFileName(yamlUrl)));
            }
        }

        private static async Task DownloadFileAsync(string url, string destinationPath)
        {
            // Skip if already present
            if (File.Exists(destinationPath))
            {
                return;
            }

            var httpClient = new HttpClient();
            using var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            if (!response.IsSuccessStatusCode)
            {
                return;
            }

            await using var stream = await response.Content.ReadAsStreamAsync();
            await using var fileStream = new FileStream(destinationPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            await stream.CopyToAsync(fileStream);
        }
    }
}
