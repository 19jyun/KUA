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

function fetchReservations() {
    const tableBody = document.getElementById('reservationsTable')?.querySelector('tbody');
    
    if (!tableBody) return;

    fetch('/reservations')
        .then(response => response.json())
        .then(data => {
            tableBody.innerHTML = ''; 
            
            data.forEach(reservation => {
                const row = tableBody.insertRow();
                
                row.insertCell().textContent = reservation.date;
                row.insertCell().textContent = reservation.time;
                row.insertCell().textContent = reservation.trainNo;
                row.insertCell().textContent = reservation.destination;

                const deleteButton = document.createElement('button');
                deleteButton.textContent = 'Delete';
                deleteButton.onclick = () => deleteReservation(reservation.id);
                row.insertCell().appendChild(deleteButton);
            });
        })
        .catch(error => console.error('Error loading reservations:', error));
}

function handleReservationSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const reservation = {
        date: formData.get('date'),
        time: formData.get('time'),
        trainNo: formData.get('trainNo'),
        destination: formData.get('destination')
    };
    
    fetch('/reservations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reservation)
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        event.target.reset();
    })
    .catch(error => console.error('Error adding reservation:', error));
}

async function deleteReservation(id) {
    try {
        const response = await fetch(`/reservations/${id}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (response.ok) {
            alert(data.message);
            fetchReservations(); 
        } else {
            alert(`Error: ${data.message}`);
        }
    } catch (error) {
        console.error('Error deleting reservation:', error);
        alert('Failed to delete reservation.');
    }
}

function goBack() {
    window.location.href = 'index.html';
}
