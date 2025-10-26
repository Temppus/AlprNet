using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace AlprNet.Lib.PlateDetection;

public class LicensePlateDetector : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int _imgSize;
    private readonly float _confThreshold;

    private bool _disposed;

    public LicensePlateDetector(
        string modelPath,
        SessionOptions? sessionOptions = null,
        float confThreshold = 0.25f)
    {
        if (modelPath == null) throw new ArgumentNullException(nameof(modelPath));
        var modelFullPath = Path.GetFullPath(modelPath);
        if (!File.Exists(modelFullPath)) throw new FileNotFoundException($"Model not found: {modelFullPath}");

        _session = new InferenceSession(modelPath, sessionOptions ?? new SessionOptions());
        _confThreshold = confThreshold;

        _inputName = _session.InputMetadata.Keys.First();

        // assume square model input
        var shape = _session.InputMetadata[_inputName].Dimensions;
        _imgSize = shape[2];
    }

    public List<LicensePlateDetectionResult> Run(Mat image)
    {
        if (image == null) { throw new ArgumentNullException(nameof(image)); }

        if (image.Empty())
        {
            throw new ArgumentException("Input image is empty");
        }

        var (inputTensor, ratio, pad) = Preprocess(image);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        using var results = _session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        return Postprocess(output, ratio, pad, image.Size());
    }

    private (DenseTensor<float> tensor, float ratio, (float dw, float dh) pad) Preprocess(Mat img)
    {
        var (resized, ratio, pad) = Letterbox(img, _imgSize);

        using (resized)
        {
            Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);
            resized.ConvertTo(resized, MatType.CV_32FC3, 1.0 / 255.0);

            var chw = new float[1 * 3 * _imgSize * _imgSize];
            var index = 0;

            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < _imgSize; y++)
                {
                    for (int x = 0; x < _imgSize; x++)
                    {
                        chw[index++] = resized.At<Vec3f>(y, x)[c];
                    }
                }
            }

            var tensor = new DenseTensor<float>(chw, new[] { 1, 3, _imgSize, _imgSize });
            return (tensor, ratio, pad);
        }
    }

    private static (Mat resized, float ratio, (float dw, float dh) pad) Letterbox(Mat src, int newSize)
    {
        int w = src.Width, h = src.Height;
        float r = Math.Min((float)newSize / w, (float)newSize / h);

        int newW = (int)(w * r);
        int newH = (int)(h * r);
        int dw = (newSize - newW) / 2;
        int dh = (newSize - newH) / 2;

        var resized = new Mat();
        Cv2.Resize(src, resized, new Size(newW, newH));
        Cv2.CopyMakeBorder(resized, resized, dh, dh, dw, dw, BorderTypes.Constant, Scalar.All(114));

        return (resized, r, (dw, dh));
    }

    private List<LicensePlateDetectionResult> Postprocess(float[] preds, float ratio, (float dw, float dh) pad, Size origSize)
    {
        // Simplified: assumes output of shape [N,7]: [id, x1, y1, x2, y2, class_id, score]
        var results = new List<LicensePlateDetectionResult>();
        int stride = 7;

        for (int i = 0; i < preds.Length; i += stride)
        {
            float score = preds[i + 6];
            if (score < _confThreshold) continue;

            float x1 = (preds[i + 1] - pad.dw) / ratio;
            float y1 = (preds[i + 2] - pad.dh) / ratio;
            float x2 = (preds[i + 3] - pad.dw) / ratio;
            float y2 = (preds[i + 4] - pad.dh) / ratio;
            int classId = (int)preds[i + 5];

            x1 = Math.Clamp(x1, 0, origSize.Width - 1);
            y1 = Math.Clamp(y1, 0, origSize.Height - 1);
            x2 = Math.Clamp(x2, 0, origSize.Width - 1);
            y2 = Math.Clamp(y2, 0, origSize.Height - 1);

            var rect = new Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));

            results.Add(new LicensePlateDetectionResult
            {
                Label = classId.ToString(),
                Confidence = score,
                BoundingBox = rect
            });
        }

        return results;
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