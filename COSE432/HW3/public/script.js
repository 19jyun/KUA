// Load and display reservations on the "My Reservations" page
document.addEventListener('DOMContentLoaded', () => {
    const tableBody = document.getElementById('reservationsTable')?.querySelector('tbody');
    
    if (tableBody) {
        fetchReservations(tableBody);
    }

    const reservationForm = document.getElementById('reservationForm');
    if (reservationForm) {
        reservationForm.addEventListener('submit', handleReservationSubmit);
    }
});

// Function to fetch reservations from the server and populate the table
function fetchReservations(tableBody) {
    fetch('/reservations')
        .then(response => response.json())
        .then(data => {
            tableBody.innerHTML = ''; // Clear any existing rows
            
            data.forEach(reservation => {
                const row = tableBody.insertRow();
                
                // Populate each cell in the row with reservation data
                row.insertCell().textContent = reservation.date;
                row.insertCell().textContent = reservation.time;
                row.insertCell().textContent = reservation.trainNo;
                row.insertCell().textContent = reservation.destination;

                // Add a delete button for each reservation
                const deleteButton = document.createElement('button');
                deleteButton.textContent = 'Delete';
                deleteButton.onclick = () => deleteReservation(reservation.id); // Use ID for deletion
                row.insertCell().appendChild(deleteButton);
            });
        })
        .catch(error => console.error('Error loading reservations:', error));
}

// Handle submission of a new reservation in "reservation.html"
function handleReservationSubmit(event) {
    event.preventDefault();
    
    // Get form data
    const formData = new FormData(event.target);
    const reservation = {
        date: formData.get('date'),
        time: formData.get('time'),
        trainNo: formData.get('trainNo'),
        destination: formData.get('destination')
    };
    
    // Send POST request to add new reservation
    fetch('/reservations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reservation)
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        event.target.reset(); // Clear the form after submission
    })
    .catch(error => console.error('Error adding reservation:', error));
}

// Function to delete a reservation by ID
function deleteReservation(id) {
    fetch(`/reservations/${id}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        const tableBody = document.getElementById('reservationsTable').querySelector('tbody');
        fetchReservations(tableBody); // Refresh the table after deletion
    })
    .catch(error => console.error('Error deleting reservation:', error));
}

// Function to go back to the main menu (if needed in other HTML files)
function goBack() {
    window.location.href = 'index.html';
}
