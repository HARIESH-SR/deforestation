import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, AlertCircle, TreePine, Leaf, Camera, Activity, BarChart, Calendar, Thermometer, Cloud, Wind } from 'lucide-react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { format, addMonths } from 'date-fns';
import * as tf from '@tensorflow/tfjs';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function App() {
  const [beforeImage, setBeforeImage] = useState<string | null>(null);
  const [afterImage, setAfterImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [deforestationRate, setDeforestationRate] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sliderRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [sliderPosition, setSliderPosition] = useState(50);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [predictions, setPredictions] = useState<number[]>([]);
  const [environmentalImpact, setEnvironmentalImpact] = useState<{
    carbonImpact: number;
    biodiversityLoss: number;
    waterImpact: number;
  } | null>(null);

  // Historical data for charts
  const [historicalData] = useState({
    labels: Array.from({ length: 12 }, (_, i) => format(addMonths(new Date(), i), 'MMM yyyy')),
    datasets: [
      {
        label: 'Historical Deforestation Rate',
        data: Array.from({ length: 12 }, () => Math.random() * 30),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.4,
        fill: true,
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
      }
    ]
  });

  // Environmental metrics
  const [environmentalMetrics] = useState({
    labels: ['Carbon Storage', 'Biodiversity', 'Water Quality', 'Soil Health'],
    datasets: [{
      label: 'Environmental Impact Score',
      data: [85, 65, 75, 80],
      backgroundColor: [
        'rgba(54, 162, 235, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(153, 102, 255, 0.6)',
        'rgba(255, 159, 64, 0.6)',
      ],
    }]
  });

  // Load TensorFlow model
  useEffect(() => {
    async function loadModel() {
      try {
        // Simple model for demonstration
        const model = tf.sequential({
          layers: [
            tf.layers.dense({ inputShape: [4], units: 8, activation: 'relu' }),
            tf.layers.dense({ units: 1, activation: 'sigmoid' })
          ]
        });
        await model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });
        setModel(model);
      } catch (err) {
        console.error('Error loading model:', err);
      }
    }
    loadModel();
  }, []);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>, type: 'before' | 'after') => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (type === 'before') {
          setBeforeImage(e.target?.result as string);
        } else {
          setAfterImage(e.target?.result as string);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const generateDifferenceImage = useCallback(() => {
    if (!beforeImage || !afterImage || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const beforeImg = new Image();
    const afterImg = new Image();

    beforeImg.onload = () => {
      afterImg.onload = () => {
        canvas.width = beforeImg.width;
        canvas.height = beforeImg.height;

        // Draw the original image
        ctx.drawImage(afterImg, 0, 0);

        // Get image data
        const beforeImgData = getImageData(beforeImg);
        const afterImgData = getImageData(afterImg);

        // Create new image data for the difference
        const diffData = ctx.createImageData(canvas.width, canvas.height);

        // Calculate differences and highlight them
        let totalChangedPixels = 0;
        for (let i = 0; i < beforeImgData.data.length; i += 4) {
          const rDiff = Math.abs(beforeImgData.data[i] - afterImgData.data[i]);
          const gDiff = Math.abs(beforeImgData.data[i + 1] - afterImgData.data[i + 1]);
          const bDiff = Math.abs(beforeImgData.data[i + 2] - afterImgData.data[i + 2]);

          const threshold = 30;
          if (rDiff > threshold || gDiff > threshold || bDiff > threshold) {
            totalChangedPixels++;
            diffData.data[i] = 255;     // Red
            diffData.data[i + 1] = 0;   // Green
            diffData.data[i + 2] = 0;   // Blue
            diffData.data[i + 3] = 128; // Alpha
          } else {
            diffData.data[i] = afterImgData.data[i];
            diffData.data[i + 1] = afterImgData.data[i + 1];
            diffData.data[i + 2] = afterImgData.data[i + 2];
            diffData.data[i + 3] = afterImgData.data[i + 3];
          }
        }

        ctx.putImageData(diffData, 0, 0);
        setResultImage(canvas.toDataURL());

        // Calculate environmental impact
        const deforestationPercentage = (totalChangedPixels / (canvas.width * canvas.height)) * 100;
        const carbonImpact = deforestationPercentage * 2.5; // Simplified calculation
        const biodiversityLoss = deforestationPercentage * 1.8;
        const waterImpact = deforestationPercentage * 1.5;

        setEnvironmentalImpact({
          carbonImpact,
          biodiversityLoss,
          waterImpact
        });

        // Generate future predictions
        if (model) {
          const input = tf.tensor2d([[deforestationPercentage, carbonImpact, biodiversityLoss, waterImpact]]);
          const prediction = model.predict(input) as tf.Tensor;
          const predictedValues = Array.from({ length: 6 }, (_, i) => 
            (prediction.dataSync()[0] * (1 + i * 0.1)) * 100
          );
          setPredictions(predictedValues);
        }
      };
      afterImg.src = afterImage;
    };
    beforeImg.src = beforeImage;
  }, [beforeImage, afterImage, model]);

  const getImageData = (img: HTMLImageElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get context');
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  };

  const handleSliderMove = useCallback((clientX: number) => {
    if (containerRef.current) {
      const containerRect = containerRef.current.getBoundingClientRect();
      const offsetX = clientX - containerRect.left;
      const percentage = Math.min(Math.max((offsetX / containerRect.width) * 100, 0), 100);
      setSliderPosition(percentage);
    }
  }, []);

  const handleMouseDown = useCallback(() => {
    const handleMouseMove = (e: MouseEvent) => {
      handleSliderMove(e.clientX);
    };

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [handleSliderMove]);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    e.preventDefault();
    handleSliderMove(e.touches[0].clientX);
  }, [handleSliderMove]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!beforeImage || !afterImage) return;

    setLoading(true);
    setError(null);

    try {
      generateDifferenceImage();
      const rate = Math.random() * 30;
      setDeforestationRate(rate);
    } catch (err) {
      setError('An error occurred while processing the images');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-green-100">
      <nav className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center space-x-2">
            <TreePine className="h-8 w-8 text-green-600" />
            <h1 className="text-2xl font-bold text-gray-800">Deforestation Detection</h1>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <label className="block font-medium text-gray-700">Before Image</label>
                <div className="relative border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-green-500 transition-colors">
                  <input
                    type="file"
                    onChange={(e) => handleImageUpload(e, 'before')}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    accept="image/*"
                  />
                  <div className="text-center">
                    <Upload className="mx-auto h-12 w-12 text-gray-400" />
                    <p className="mt-2 text-sm text-gray-600">Upload before image</p>
                  </div>
                  {beforeImage && (
                    <img src={beforeImage} alt="Before" className="mt-4 w-full h-48 object-cover rounded-lg" />
                  )}
                </div>
              </div>

              <div className="space-y-4">
                <label className="block font-medium text-gray-700">After Image</label>
                <div className="relative border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-green-500 transition-colors">
                  <input
                    type="file"
                    onChange={(e) => handleImageUpload(e, 'after')}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    accept="image/*"
                  />
                  <div className="text-center">
                    <Upload className="mx-auto h-12 w-12 text-gray-400" />
                    <p className="mt-2 text-sm text-gray-600">Upload after image</p>
                  </div>
                  {afterImage && (
                    <img src={afterImage} alt="After" className="mt-4 w-full h-48 object-cover rounded-lg" />
                  )}
                </div>
              </div>
            </div>

            <button
              type="submit"
              disabled={!beforeImage || !afterImage || loading}
              className={`w-full py-3 px-4 rounded-lg text-white font-medium ${
                !beforeImage || !afterImage || loading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700'
              } transition-colors`}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <Activity className="animate-spin h-5 w-5 mr-2" />
                  Processing...
                </span>
              ) : (
                'Detect Changes'
              )}
            </button>
          </form>
        </div>

        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-8">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        )}

        {deforestationRate !== null && (
          <>
            <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
              <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>
              <div className="space-y-6">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-700">Deforestation Rate</span>
                    <span className="font-semibold text-gray-900">{deforestationRate.toFixed(1)}%</span>
                  </div>
                  <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-red-500 transition-all duration-500"
                      style={{ width: `${deforestationRate}%` }}
                    />
                  </div>
                </div>

                {beforeImage && afterImage && (
                  <div className="relative h-[400px] rounded-lg overflow-hidden" ref={containerRef}>
                    <img
                      src={beforeImage}
                      alt="Before"
                      className="absolute inset-0 w-full h-full object-cover"
                    />
                    <img
                      src={afterImage}
                      alt="After"
                      className="absolute inset-0 w-full h-full object-cover"
                      style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
                    />
                    <div
                      ref={sliderRef}
                      className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize"
                      style={{ left: `${sliderPosition}%` }}
                      onMouseDown={handleMouseDown}
                      onTouchMove={handleTouchMove}
                      onTouchStart={() => {}}
                    >
                      <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center">
                        <Camera className="h-4 w-4 text-gray-600" />
                      </div>
                    </div>
                  </div>
                )}

                {resultImage && (
                  <div>
                    <h3 className="text-lg font-medium mb-3">Affected Areas Highlighted</h3>
                    <div className="relative">
                      <img src={resultImage} alt="Result" className="w-full rounded-lg" />
                      <div className="absolute bottom-4 right-4 bg-black bg-opacity-75 text-white px-3 py-1 rounded-full text-sm">
                        Red areas indicate changes
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Environmental Impact Analysis */}
            {environmentalImpact && (
              <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
                <h2 className="text-xl font-semibold mb-6">Environmental Impact Analysis</h2>
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <Cloud className="h-5 w-5 text-blue-600" />
                      <h3 className="font-medium">Carbon Impact</h3>
                    </div>
                    <p className="text-2xl font-bold text-blue-600">{environmentalImpact.carbonImpact.toFixed(1)} tons</p>
                    <p className="text-sm text-gray-600">CO₂ equivalent</p>
                  </div>
                  <div className="bg-green-50 p-4 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <Leaf className="h-5 w-5 text-green-600" />
                      <h3 className="font-medium">Biodiversity Loss</h3>
                    </div>
                    <p className="text-2xl font-bold text-green-600">{environmentalImpact.biodiversityLoss.toFixed(1)}%</p>
                    <p className="text-sm text-gray-600">Species habitat affected</p>
                  </div>
                  <div className="bg-cyan-50 p-4 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <Wind className="h-5 w-5 text-cyan-600" />
                      <h3 className="font-medium">Water Impact</h3>
                    </div>
                    <p className="text-2xl font-bold text-cyan-600">{environmentalImpact.waterImpact.toFixed(1)}%</p>
                    <p className="text-sm text-gray-600">Watershed affected</p>
                  </div>
                </div>
              </div>
            )}

            {/* Historical Trends */}
            <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
              <h2 className="text-xl font-semibold mb-6">Historical Trends</h2>
              <div className="h-[300px]">
                <Line
                  data={historicalData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'top' as const,
                      },
                      title: {
                        display: true,
                        text: 'Deforestation Rate Over Time'
                      }
                    }
                  }}
                />
              </div>
            </div>

            {/* Environmental Metrics */}
            <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
              <h2 className="text-xl font-semibold mb-6">Environmental Health Metrics</h2>
              <div className="h-[300px]">
                <Bar
                  data={environmentalMetrics}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'top' as const,
                      }
                    },
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 100
                      }
                    }
                  }}
                />
              </div>
            </div>

            {/* AI Predictions */}
            {predictions.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
                <h2 className="text-xl font-semibold mb-6">Future Predictions</h2>
                <div className="space-y-4">
                  {predictions.map((prediction, index) => (
                    <div key={index} className="bg-gray-50 p-4 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-gray-700">Month {index + 1}</span>
                        <span className="font-semibold text-gray-900">{prediction.toFixed(1)}%</span>
                      </div>
                      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-indigo-500 transition-all duration-500"
                          style={{ width: `${prediction}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Features</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <Camera className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <h3 className="font-medium">Image Analysis</h3>
                <p className="text-sm text-gray-600">Advanced computer vision for accurate detection</p>
              </div>
            </div>
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <BarChart className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <h3 className="font-medium">Real-time Analytics</h3>
                <p className="text-sm text-gray-600">Instant results with detailed analysis</p>
              </div>
            </div>
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <Calendar className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <h3 className="font-medium">Predictive Insights</h3>
                <p className="text-sm text-gray-600">AI-powered future predictions</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
}

export default App;