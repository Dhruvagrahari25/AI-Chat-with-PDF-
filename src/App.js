import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [content, setContent] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    if (data.filename) {
      const displayResponse = await fetch(`http://localhost:5000/display/${data.filename}`);
      const displayData = await displayResponse.json();
      setContent(displayData.content);
    }
  };

  return (
    <div className="App">
      <h1>Upload PDF</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      <h2>PDF Contents</h2>
      <pre>{content}</pre>
    </div>
  );
}

export default App;
