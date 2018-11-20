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
            var data = reader.Read(dataPath);

            // Build the training pipeline.
            var pipeline =
                // Concatenate all the features together into one column 'Features'.
                mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                // Note that the label is text, so it needs to be converted to key.
                .Append(mlContext.Transforms.Categorical.MapValueToKey("Label"), TransformerScope.TrainTest)
                // Use the multi-class SDCA model to predict the label using features.
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent())
                // Apply the inverse conversion from 'PredictedLabel' column back to string value.
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(("PredictedLabel", "Data")));

            var model = pipeline.Fit(data);

            var predictionEngine = model.MakePredictionFunction<IrisInput, IrisPrediction>(mlContext);
            var prediction = predictionEngine.Predict(new IrisInput
            {
                SepalLength = 4.1f,
                SepalWidth = 0.1f,
                PetalLength = 3.2f,
                PetalWidth = 1.4f
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabel}");
        }
    }
}
