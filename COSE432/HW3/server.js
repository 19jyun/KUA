const express = require('express');
const fs = require('fs');
const path = require('path');
const bcrypt = require('bcryptjs');
const session = require('express-session');
const { v4: uuidv4 } = require('uuid');
const app = express();
const PORT = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Initialize sessions
app.use(session({
    secret: 'your_secret_key',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false } // Set secure: true in production (with HTTPS)
}));

function ensureSignupFileExists() {
    const filePath = path.join(__dirname, 'signup_info.csv');
    if (!fs.existsSync(filePath)) {
        fs.writeFileSync(filePath, 'userID,hashedPassword\n');
        console.log('Created signup_info.csv with headers');
    }
}

function ensureReservationsFileExists() {
    const filePath = path.join(__dirname, 'reservations.csv');
    if (!fs.existsSync(filePath)) {
        fs.writeFileSync(filePath, 'id,userID,year,month,day,time,trainNo,destination\n');
        console.log('Created reservations.csv with headers');
    }
}

// Call helper functions at startup to ensure files exist
ensureSignupFileExists();
ensureReservationsFileExists();

// Helper function to get month abbreviation
function getMonthAbbreviation(month) {
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return months[parseInt(month, 10) - 1];
}

// Middleware to check if user is authenticated
function isAuthenticated(req, res, next) {
    if (req.session.userID) {
        return next();
    }
    res.status(401).json({ message: 'Unauthorized' });
}

// Endpoint to get the logged-in user's info
app.get('/user-info', isAuthenticated, (req, res) => {
    res.json({ userID: req.session.userID });
});

// Signup endpoint
app.post('/signup', async (req, res) => {
    const { userID, password } = req.body;
    ensureSignupFileExists(); // Ensure the signup file exists

    // Check if userID already exists
    const data = fs.readFileSync('signup_info.csv', 'utf8');
    const users = data.trim().split('\n').slice(1).map(line => line.split(',')[0]);
    if (users.includes(userID)) {
        return res.status(400).json({ message: 'UserID already exists' });
    }

    // Hash the password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Append the new user to signup_info.csv
    const newUser = `\n${userID},${hashedPassword}`;
    fs.appendFileSync('signup_info.csv', newUser);
    res.json({ message: 'Signup successful' });
});

// Login endpoint
app.post('/login', async (req, res) => {
    const { userID, password } = req.body;
    ensureSignupFileExists(); // Ensure the signup file exists

    const data = fs.readFileSync('signup_info.csv', 'utf8');
    const user = data.trim().split('\n').slice(1).find(line => line.split(',')[0] === userID);

    if (!user) {
        return res.status(400).json({ message: 'Invalid credentials' });
    }

    const [storedUserID, storedHashedPassword] = user.split(',');

    // Verify password
    const isMatch = await bcrypt.compare(password, storedHashedPassword);
    if (!isMatch) {
        return res.status(400).json({ message: 'Invalid credentials' });
    }

    // Start session
    req.session.userID = userID;
    res.json({ message: `Login successful! Welcome, ${userID}` });
});

// Logout endpoint
app.post('/logout', (req, res) => {
    req.session.destroy((err) => {
        if (err) {
            return res.status(500).json({ message: 'Error logging out' });
        }
        res.json({ message: 'Logout successful' });
    });
});

// Check if userID is available (for signup)
app.get('/check-userid/:userID', (req, res) => {
    const userIDToCheck = req.params.userID;
    ensureSignupFileExists(); // Ensure the signup file exists

    const data = fs.readFileSync('signup_info.csv', 'utf8');
    const users = data.trim().split('\n').slice(1).map(line => line.split(',')[0]);
    if (users.includes(userIDToCheck)) {
        return res.status(400).json({ message: 'UserID is taken' });
    }

    res.json({ message: 'UserID is available' });
});

// Endpoint to add a new reservation (protected by isAuthenticated middleware)
app.post('/reservations', isAuthenticated, (req, res) => {
    ensureReservationsFileExists(); // Ensure the reservations file exists

    const { date, time, trainNo, destination } = req.body;
    const userID = req.session.userID; // Get the logged-in user's ID

    if (!date || !time || !trainNo || !destination) {
        return res.status(400).json({ message: 'Missing required reservation data' });
    }

    const [year, month, day] = date.split('-');
    const monthAbbreviation = getMonthAbbreviation(month);
    const newId = uuidv4(); // Generate a unique ID for the reservation

    // Format the new reservation with the userID
    const newReservation = `\n${newId},${userID},${year},${monthAbbreviation},${day},${time},${trainNo},${destination}`;
    fs.appendFileSync('reservations.csv', newReservation);
    res.json({ message: 'Reservation added successfully', id: newId });
});

// Endpoint to get reservations for the logged-in user
app.get('/reservations', isAuthenticated, (req, res) => {
    ensureReservationsFileExists(); // Ensure the reservations file exists

    const userID = req.session.userID; // Get the logged-in user's ID
    const data = fs.readFileSync('reservations.csv', 'utf8');

    const rows = data.trim().split('\n').slice(1);
    const reservations = rows
        .map(line => {
            const [id, storedUserID, year, month, day, time, trainNo, destination] = line.split(',').map(item => item.trim());
            if (storedUserID === userID) {
                return { id, date: `${year}-${month}-${day}`, time, trainNo, destination };
            }
        })
        .filter(Boolean);

    res.json(reservations);
});

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
