// frontend/src/components/UploadForm.jsx
import React, { useState } from 'react';

const UploadForm = () => {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [modelType, setModelType] = useState('tree_species');

  const handleFileChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleModelTypeChange = (e) => {
    setModelType(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append('image', image);
    formData.append('model_type', modelType);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      setPrediction(result.prediction);
    } catch (error) {
      console.error("Error uploading image:", error);
    }
  };

  return (
    <div>
      <h2>Upload Plant or Tree Image</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <input type="file" onChange={handleFileChange} />
        </div>
        <div>
          <select onChange={handleModelTypeChange}>
            <option value="tree_species">Tree Species</option>
            <option value="plant_disease">Plant Disease</option>
          </select>
        </div>
        <button type="submit">Submit</button>
      </form>

      {prediction && <p>Prediction: {prediction}</p>}
    </div>
  );
};

export default UploadForm;

