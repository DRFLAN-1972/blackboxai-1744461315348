document.getElementById('proceed').addEventListener('click', function() {
    const email = document.getElementById('email').value;
    const ownerEmails = [
        "rjflanary1@gmail.com",
        "flanard72@gmail.com",
        "tflanary713@gmail.com",
        "babyallen06@yahoo.com",
        "iamliterallybetter@gmail.com",
        "flanaryray@outlook.com"
    ];

    if (email === "") {
        alert("Please enter your email.");
        return;
    }

    if (ownerEmails.includes(email)) {
        // Redirect to app.html for owner access
        window.location.href = "app.html";
    } else {
        // Show terms popup
        document.getElementById('terms-popup').classList.remove('hidden');
    }
});

document.getElementById('accept-terms').addEventListener('click', function() {
    // Redirect to checkout.html for non-owner access
    window.location.href = "checkout.html";
});
