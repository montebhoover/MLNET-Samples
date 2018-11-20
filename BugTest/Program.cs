using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Conversions;
using System;

namespace StaticSdca
{
    class Program
    {
        static void Main(string[] args)
        {
            // STEP 1: Define your data structures (in IrisDataSchema.cs)

            // STEP 2: Create a pipeline and load your data

            // 0.6.0 API
            var mlContext = new MLContext();

            // If working in Visual Studio, make sure the 'Copy to Output Directory' 
            // property of iris-data.txt is set to 'Copy always'
            string dataPath = "iris-data.txt";

            // 0.6.0 API - static
            var reader = TextLoader.CreateReader(mlContext, ctx => (
                    SepalLength: ctx.LoadFloat(0),
                    SepalWidth: ctx.LoadFloat(1),
                    PetalLength: ctx.LoadFloat(2),
                    PetalWidth: ctx.LoadFloat(3),
                    // Optionally load all columns between 0 and 3 as a single "Features" column that contains vectors of floats
                    //Features: ctx.LoadFloat(0, 3),
                    Label: ctx.LoadText(4)
                ),
                // Default separator is tab, but the dataset has comma.
                separator: ',');

            // 0.6.0 API - static
            var trainingPipeline = reader.MakeNewEstimator()
                // Add "ConcatWith" transformer that produces a new column called "Features"
                .Append(row => (
                    row.Label,
                    Features: row.SepalLength.ConcatWith(row.SepalWidth, row.PetalLength, row.PetalWidth)))
                // Add SDCA trainer that produces a new column called "Predictions"
                .Append(row => (
                    row.Label,
                    Predictions: mlContext.MulticlassClassification.Trainers.Sdca(row.Label.ToKey(), row.Features)))
                // Add "ToValue" transformer that converts labels back to strings
                .Append(row => row.Predictions.predictedLabel.ToValue());

            var data = reader.Read(new MultiFileSource(dataPath));
            var model = trainingPipeline.Fit(data).AsDynamic;

            var predictionEngine = model.MakePredictionFunction<IrisInput, IrisPrediction>(mlContext);
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


            // STEP 4: Add learner
            // Add a learning algorithm to the pipeline. 
            // This is a classification scenario (What type of iris is this?)


            // STEP 5: Train your model based on the data set


            // STEP 6: Use your model to make a prediction
            // You can change these numbers to test different predictions



            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabel}");
        }
    }
}
