const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS
app.use(cors());

// Serve static files
app.use(express.static(path.join(__dirname)));

// Serve the main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'Frontend server is running' });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 StudyMate Frontend Server Started!`);
    console.log(`📱 Access the app at: http://localhost:${PORT}`);
    console.log(`📄 Make sure your Python backend is running on port 5000`);
    console.log(`⚠️  Press Ctrl+C to stop the server`);
});