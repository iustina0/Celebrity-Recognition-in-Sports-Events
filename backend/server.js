const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const multer = require('multer');

const app = express();
const port = 3000;

app.use(bodyParser.json());
const upload = multer({ dest: 'uploads/' });
app.use(cors());


app.post('/api/process-video', upload.single('video'), (req, res) => {
    const videoPath = req.file.path;

    const pythonProcess = spawn('python', ['./backend/bridge.py', videoPath]);

    pythonProcess.stdout.on('data', (data) => {
        res.send(data);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error: ${data}`);
    });
});



app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});