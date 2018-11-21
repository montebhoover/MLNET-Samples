using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace StaticSdca
{
    class IrisInput
    {
        public float SepalLength { get; set; }
        public float SepalWidth { get; set; }
        public float PetalLength { get; set; }
        public float PetalWidth { get; set; }
        // Unfortunately, we still need the dummy 'Label' column to be present.
        [ColumnName("Label")]
        public string IgnoredLabel { get; set; }
    }

    class IrisPrediction
    {
        [ColumnName("Data")]
        public string PredictedLabel { get; set; }
    }
}
