const express = require('express');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid'); // Importing UUID for unique IDs
const app = express();
const PORT = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Helper function to ensure `reservations.csv` exists with the correct headers
function ensureCSVFileExists() {
    const filePath = path.join(__dirname, 'reservations.csv');
    if (!fs.existsSync(filePath)) {
        // Create the file with headers if it doesn't exist
        fs.writeFileSync(filePath, 'id,year,month,day,time,trainNo,destination\n');
        console.log('Created reservations.csv with headers');
    }
}

ensureCSVFileExists();

// Helper function to convert numeric month to abbreviated month name
function getMonthAbbreviation(month) {
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return months[parseInt(month, 10) - 1];
}

// Endpoint to add a new reservation with a unique ID
app.post('/reservations', (req, res) => {
    const { date, time, trainNo, destination } = req.body;

    if (!date || !time || !trainNo || !destination) {
        return res.status(400).json({ message: 'Missing required reservation data' });
    }

    const [year, month, day] = date.split('-');
    const monthAbbreviation = getMonthAbbreviation(month);
    const newId = uuidv4(); // Generate a unique ID for the reservation

    // Format the new reservation with the ID and the month abbreviation
    const newReservation = `\n${newId}, ${year}, ${monthAbbreviation}, ${day}, ${time}, ${trainNo}, ${destination}`;
    
    // Append the new reservation to reservations.csv
    fs.appendFile('reservations.csv', newReservation, (err) => {
        if (err) {
            console.error('Error saving reservation:', err);
            return res.status(500).json({ message: 'Error saving reservation' });
        }
        res.json({ message: 'Reservation added successfully', id: newId });
    });
});

// Endpoint to get all reservations from the CSV file
app.get('/reservations', (req, res) => {
    fs.readFile('reservations.csv', 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading reservations file:', err);
            return res.status(500).json({ message: 'Error reading reservations file' });
        }

        // Split the CSV file into rows and parse each row into an object
        const rows = data.trim().split('\n');
        const header = rows[0];
        const dataRows = rows.slice(1);

        const reservations = dataRows.map(line => {
            const [id, year, month, day, time, trainNo, destination] = line.split(',').map(item => item.trim());
            return {
                id,
                date: `${year}-${month}-${day}`,
                time,
                trainNo,
                destination
            };
        });

        res.json(reservations);
    });
});

// Endpoint to delete a specific reservation by ID
app.delete('/reservations/:id', (req, res) => {
    const idToDelete = req.params.id;

    fs.readFile('reservations.csv', 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading reservations file:', err);
            return res.status(500).json({ message: 'Error reading reservations file' });
        }

        const rows = data.trim().split('\n');
        const header = rows[0];
        const dataRows = rows.slice(1);

        let found = false;

        // Filter out the row that matches the id to delete
        const remainingRows = dataRows.filter(row => {
            const [rowId] = row.split(',').map(item => item.trim());
            if (rowId === idToDelete) {
                found = true;
                return false; // Exclude this row
            }
            return true; // Keep all other rows
        });

        if (!found) {
            return res.status(404).json({ message: 'Reservation not found' });
        }

        // Join the header and remaining rows and write back to CSV
        const updatedData = [header, ...remainingRows].join('\n');
        fs.writeFile('reservations.csv', updatedData, (err) => {
            if (err) {
                console.error('Error writing to reservations file:', err);
                return res.status(500).json({ message: 'Error deleting reservation' });
            }
            res.json({ message: 'Reservation deleted successfully' });
        });
    });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
