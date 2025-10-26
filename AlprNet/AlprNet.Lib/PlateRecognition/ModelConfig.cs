using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace AlprNet.Lib.PlateRecognition
{
    public enum ModelImageColorMode
    {
        Grayscale,
        Rgb
    }

    public enum ModelImageInterpolation
    {
        Linear,
        Nearest
    }

    public record PlateOCRConfig
    {
        public int MaxPlateSlots { get; init; }
        public string Alphabet { get; init; } = string.Empty;
        public string PadChar { get; init; } = string.Empty;
        public int ImgHeight { get; init; }
        public int ImgWidth { get; init; }
        public bool KeepAspectRatio { get; init; }
        public ModelImageInterpolation Interpolation { get; init; } = ModelImageInterpolation.Linear;
        public ModelImageColorMode ImageColorMode { get; init; } = ModelImageColorMode.Grayscale;

        public const int DefaultPaddingColor = 114;
        public object PaddingColor { get; init; } = DefaultPaddingColor;

        public int VocabularySize => Alphabet.Length;
        public int NumChannels => ImageColorMode == ModelImageColorMode.Rgb ? 3 : 1;

        public static PlateOCRConfig FromYaml(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Config file not found: {path}");

            var yaml = File.ReadAllText(path);

            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .IgnoreUnmatchedProperties()
                .Build();

            var rawData = deserializer.Deserialize<Dictionary<string, object>>(yaml);
            return FromDictionary(rawData);
        }

        private static PlateOCRConfig FromDictionary(Dictionary<string, object> data)
        {
            var colorMode = Get("image_color_mode", ModelImageColorMode.Grayscale);
            var config = new PlateOCRConfig
            {
                MaxPlateSlots = Get("max_plate_slots", 0),
                Alphabet = Get("alphabet", string.Empty),
                PadChar = Get("pad_char", string.Empty),
                ImgHeight = Get("img_height", 0),
                ImgWidth = Get("img_width", 0),
                KeepAspectRatio = Get("keep_aspect_ratio", false),
                Interpolation = Get("interpolation", ModelImageInterpolation.Linear),
                ImageColorMode = colorMode
            };

            // Handle padding_color: int | List<int> | null
            if (data.TryGetValue("padding_color", out var pc) && pc != null)
            {
                config = pc switch
                {
                    long intVal => config with {PaddingColor = (int) intVal},
                    int intVal => config with {PaddingColor = intVal},
                    IList<object> {Count: 3} list => config with
                    {
                        PaddingColor = (Convert.ToInt32(list[0]), Convert.ToInt32(list[1]), Convert.ToInt32(list[2]))
                    },
                    _ => config
                };
            }
            else
            {
                config = config with
                {
                    PaddingColor = colorMode == ModelImageColorMode.Rgb
                        ? (114, 114, 114)
                        : 114
                };
            }

            if (string.IsNullOrEmpty(config.Alphabet))
                throw new InvalidDataException("Alphabet cannot be empty.");
            if (string.IsNullOrEmpty(config.PadChar) || !config.Alphabet.Contains(config.PadChar))
                throw new InvalidDataException("PadChar must be in Alphabet.");

            return config;

            T Get<T>(string key, T defaultValue = default!)
            {
                if (data.TryGetValue(key, out var val) && val != null)
                {
                    try
                    {
                        if (typeof(T).IsEnum)
                            return (T)Enum.Parse(typeof(T), val.ToString()!, true);
                        return (T)Convert.ChangeType(val, typeof(T));
                    }
                    catch { /* ignore conversion issues */ }
                }
                return defaultValue;
            }
        }

        public override string ToString()
        {
            return $"PlateOCRConfig(MaxSlots={MaxPlateSlots}, AlphabetSize={VocabularySize}, " +
                   $"Img={ImgWidth}x{ImgHeight}, Mode={ImageColorMode}, Channels={NumChannels})";
        }
    }
}
