using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace AlprNet.Lib.PlateRecognition
{
    public class LicensePlateRecognizer : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly PlateOCRConfig _config;
        private bool _disposed;

        public LicensePlateRecognizer(string onnxModelPath, string plateConfigPath, SessionOptions? sessionOptions = null)
        {
            if (onnxModelPath == null) throw new ArgumentNullException(nameof(onnxModelPath));
            if (plateConfigPath == null) throw new ArgumentNullException(nameof(plateConfigPath));

            var modelPath = Path.GetFullPath(onnxModelPath);
            var configPath = Path.GetFullPath(plateConfigPath);

            if (!File.Exists(modelPath)) throw new FileNotFoundException($"Model not found: {modelPath}");
            if (!File.Exists(configPath)) throw new FileNotFoundException($"Config not found: {configPath}");

            _config = PlateOCRConfig.FromYaml(configPath);

            _session = new InferenceSession(modelPath, sessionOptions ?? new SessionOptions());
        }

        public (List<string> plates, float[,]? confidences) Run(Mat[] images, bool returnConfidence = false)
        {
            if (images == null || images.Length == 0)
                throw new ArgumentException("Input Mat array cannot be null or empty.", nameof(images));

            var batchTensor = BuildUint8Batch(images, _config);
            var logits = RunInternal(batchTensor);

            int batchSize = images.Length;
            int slots = _config.MaxPlateSlots;
            int vocab = _config.Alphabet.Length;

            var plates = new List<string>(batchSize);
            float[,]? confidences = returnConfidence ? new float[batchSize, slots] : null;

            for (int n = 0; n < batchSize; n++)
            {
                // extract per-sample logits (flattened)
                var sample = new float[vocab * slots];
                Array.Copy(logits, n * vocab * slots, sample, 0, sample.Length);

                var (plate, conf) = PostProcess(sample, _config, returnConfidence);
                plates.Add(plate);

                if (!returnConfidence || conf == null)
                {
                    continue;
                }

                for (int j = 0; j < slots; j++)
                    confidences![n, j] = conf[j];
            }

            return (plates, confidences);
        }

        private float[] RunInternal(DenseTensor<byte> inputTensor)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using var results = _session.Run(inputs);
            return results[0].AsTensor<float>().ToArray();
        }

        /// <summary>
        /// Builds a single NHWC uint8 tensor for the whole batch.
        /// </summary>
        private static DenseTensor<byte> BuildUint8Batch(Mat[] images, PlateOCRConfig cfg)
        {
            int N = images.Length;
            int H = cfg.ImgHeight;
            int W = cfg.ImgWidth;
            int C = cfg.NumChannels;

            var tensor = new DenseTensor<byte>([N, H, W, C]);

            for (int n = 0; n < N; n++)
                CopyResizedPlate(images[n], tensor, n, cfg);

            return tensor;
        }

        /// <summary>
        /// Resizes, pads and copies one plate into the pre-allocated slice of the batch tensor.
        /// </summary>
        private static void CopyResizedPlate(Mat src, DenseTensor<byte> batch, int batchIdx, PlateOCRConfig cfg)
        {
            int targetH = cfg.ImgHeight;
            int targetW = cfg.ImgWidth;
            int channels = cfg.NumChannels;

            // -------------------------------------------------------------
            // 1. Compute scale that preserves aspect ratio
            // -------------------------------------------------------------
            float scale = Math.Min((float)targetW / src.Width, (float)targetH / src.Height);
            int newW = (int)(src.Width * scale);
            int newH = (int)(src.Height * scale);

            var interpolationFlags = cfg.Interpolation switch
            {
                ModelImageInterpolation.Linear => InterpolationFlags.Linear,
                ModelImageInterpolation.Nearest => InterpolationFlags.Nearest,
                _ => InterpolationFlags.Cubic
            };

            // -------------------------------------------------------------
            // 2. Resize to the scaled size
            // -------------------------------------------------------------
            using var resized = src.Resize(new Size(newW, newH), 0, 0, interpolationFlags);

            // -------------------------------------------------------------
            // 3. Create a canvas filled with the padding colour (114,114,114)
            // -------------------------------------------------------------
            var padColor = new Scalar(114, 114, 114);
            using var canvas = new Mat(new Size(targetW, targetH),
                                      channels == 1 ? MatType.CV_8UC1 : MatType.CV_8UC3,
                                      padColor);

            // -------------------------------------------------------------
            // 4. Paste the resized image centred on the canvas
            // -------------------------------------------------------------
            int offsetX = (targetW - newW) / 2;
            int offsetY = (targetH - newH) / 2;
            var roi = new Rect(offsetX, offsetY, newW, newH);
            resized.CopyTo(canvas[roi]);

            // -------------------------------------------------------------
            // 5. Copy the canvas into the batch tensor (NHWC, uint8)
            // -------------------------------------------------------------
            if (channels == 3)
            {
                // OpenCV stores BGR → we need RGB for the model (the Python code does BGR→RGB)
                var indexer = canvas.GetGenericIndexer<Vec3b>();
                for (int y = 0; y < targetH; y++)
                    for (int x = 0; x < targetW; x++)
                    {
                        var p = indexer[y, x];
                        batch[batchIdx, y, x, 0] = p.Item2; // R
                        batch[batchIdx, y, x, 1] = p.Item1; // G
                        batch[batchIdx, y, x, 2] = p.Item0; // B
                    }
            }
            else // grayscale
            {
                for (int y = 0; y < targetH; y++)
                    for (int x = 0; x < targetW; x++)
                        batch[batchIdx, y, x, 0] = canvas.At<byte>(y, x);
            }
        }

        private static (string plate, float[]? confidences) PostProcess(
            float[] logits, PlateOCRConfig cfg, bool returnConf)
        {
            int slots = cfg.MaxPlateSlots;
            int vocab = cfg.Alphabet.Length;   // match Python's len(model_alphabet)

            var probs = Softmax(logits, vocab);

            var chars = new char[slots];
            float[]? confs = returnConf ? new float[slots] : null;

            for (int t = 0; t < slots; t++)
            {
                int start = t * vocab;
                float maxProb = float.NegativeInfinity;
                int bestIdx = 0;

                // iterate across the entire vocabulary (including any pad/blank token if present)
                for (int c = 0; c < vocab; c++)
                {
                    float p = probs[start + c];
                    if (p > maxProb)
                    {
                        maxProb = p;
                        bestIdx = c;
                    }
                }

                chars[t] = cfg.Alphabet[bestIdx];
                if (returnConf) confs![t] = maxProb;
            }

            string plate = new string(chars).Replace(cfg.PadChar, string.Empty).Trim();

            return (plate, confs);
        }

        private static float[] Softmax(float[] x, int size)
        {
            var result = new float[x.Length];

            for (int i = 0; i + size <= x.Length; i += size)
            {
                // find max for numerical stability
                float max = x[i];
                for (int j = 1; j < size; j++)
                    if (x[i + j] > max) max = x[i + j];

                float sum = 0;
                for (int j = 0; j < size; j++)
                {
                    result[i + j] = MathF.Exp(x[i + j] - max);
                    sum += result[i + j];
                }

                for (int j = 0; j < size; j++)
                    result[i + j] /= sum;
            }

            return result;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _session.Dispose();
            _disposed = true;
        }
    }
}
