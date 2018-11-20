using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;


namespace DynamicSdca
{
    class Program
    {
        static void Main(string[] args)
        {
            // STEP 1: Create a pipeline and load your data
            // 0.7.0 API
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // If working in Visual Studio, make sure the 'Copy to Output Directory' 
            // property of iris-data.txt is set to 'Copy always'
            string dataPath = "iris-data.txt";

            // 0.6.0 API - dynamic
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Column = new[] {
                    new TextLoader.Column("SepalLength", DataKind.R4, 0),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                    new TextLoader.Column("PetalLength", DataKind.R4, 2),
                    new TextLoader.Column("PetalWidth", DataKind.R4, 3),
                    new TextLoader.Column("Label", DataKind.TX, 4)
                },
                // Default separator is tab, but the dataset has comma.
                Separator = ","
            });

            var pipeline =
                // Concatenate all the features together into one column 'Features'.
                new ConcatEstimator(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                // Note that the label is text, so it needs to be converted to key.
                .Append(new ToKeyEstimator(env, "Label"))
                // Use the multi-class SDCA model to predict the label using features.
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent())
                // Apply the inverse conversion from 'PredictedLabel' column back to string value.
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(("PredictedLabel", "Data")));

            var data = reader.Read(new MultiFileSource(dataPath));
            var model = pipeline.Fit(data).AsDynamic;

            var predictionEngine = model.MakePredictionFunction<IrisInput, IrisPrediction>(env);
            var prediction = predictionEngine.Predict(new IrisInput
            {
                SepalLength = 4.1f,
                SepalWidth = 0.1f,
                PetalLength = 3.2f,
                PetalWidth = 1.4f
            });

            //// Apply all kinds of standard ML.NET normalization to the raw features.
            //var normalizerPipeline = reader.MakeNewEstimator()
            //    .Append(r => (
            //        MinMaxNormalized: r.Features.Normalize(fixZero: true),
            //        MeanVarNormalized: r.Features.NormalizeByMeanVar(fixZero: false),
            //        CdfNormalized: r.Features.NormalizeByCumulativeDistribution(),
            //        BinNormalized: r.Features.NormalizeByBinning(maxBins: 256)
            //    ));

            //// Let's train our pipeline of normalizers, and then apply it to the same data.
            //var normalizedData = normalizerPipeline.Fit(data).Transform(data);

            //// Inspect one column of the resulting dataset.
            //var minMaxValues = normalizedData.GetColumn(r => r.MinMaxNormalized).ToArray();
            //var meanVarValues = normalizedData.GetColumn(r => r.MeanVarNormalized).ToArray();
            //var cdfValues = normalizedData.GetColumn(r => r.CdfNormalized).ToArray();
            //var binValues = normalizedData.GetColumn(r => r.BinNormalized).ToArray();


            // STEP 3: Transform your data

            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training

            // Legacy API
            //pipeline.Add(new Dictionarizer("Label"));

            // Puts all features into a vector

            // Legacy API
            //pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            // STEP 4: Add learner
            // Add a learning algorithm to the pipeline. 
            // This is a classification scenario (What type of iris is this?)

            // Legacy API
            //pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // Convert the Label back into original text (after converting to number in step 3)

            // Legacy API
            //pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // STEP 5: Train your model based on the data set

            // Legacy API
            //var model = pipeline.Train<IrisDataLegacy, IrisPredictionLegacy>();

            // STEP 6: Use your model to make a prediction
            // You can change these numbers to test different predictions

            // Legacy API
            //var prediction = model.Predict(new IrisDataLegacy()
            //{
            //    SepalLength = 3.3f,
            //    SepalWidth = 1.6f,
            //    PetalLength = 0.2f,
            //    PetalWidth = 5.1f,
            //});

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabel}");
        }
    }
}
