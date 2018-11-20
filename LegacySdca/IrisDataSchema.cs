using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace LegacySdca
{
    // Legacy API
    class IrisData
    {
        [Column("0")]
        public float SepalLength;

        [Column("1")]
        public float SepalWidth;

        [Column("2")]
        public float PetalLength;

        [Column("3")]
        public float PetalWidth;

        [Column("4")]
        [ColumnName("Label")]
        public string Label;
    }

    class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel;
    }
}
