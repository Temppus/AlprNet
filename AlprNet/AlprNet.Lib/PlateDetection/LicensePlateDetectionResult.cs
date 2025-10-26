using OpenCvSharp;

namespace AlprNet.Lib.PlateDetection
{
    public class LicensePlateDetectionResult
    {
        public string Label { get; set; } = string.Empty;
        public float Confidence { get; set; }
        public Rect BoundingBox { get; set; }
    }

    public static class LicensePlateDetectionResultExtensions
    {
        public static Mat ToPlateCropImage(this LicensePlateDetectionResult detection, Mat source)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));

            if (detection is null)
                throw new ArgumentNullException(nameof(detection));

            if (source.Empty())
                throw new ArgumentException("Source image is empty.", nameof(source));

            var bbox = detection.BoundingBox;

            // Ensure the bounding box is within image bounds
            int x = Math.Max(bbox.X, 0);
            int y = Math.Max(bbox.Y, 0);
            int w = Math.Min(bbox.Width, source.Width - x);
            int h = Math.Min(bbox.Height, source.Height - y);

            // If box is invalid or too small, return empty Mat
            if (w <= 0 || h <= 0)
                return new Mat();

            var roi = new Rect(x, y, w, h);
            return new Mat(source, roi).Clone();  // clone to own memory safely
        }
    }
}
