// Package definitions with their details
const packages = {
    bronze: {
        name: 'Bronze',
        basePrice: 29.99,
        features: [
            'Up to 5-minute songs',
            '5 songs per month',
            'World-class mastering',
            'Premium effects',
            '5 voice clones'
        ],
        maxDuration: 300, // 5 minutes in seconds
        songsPerMonth: 5,
        voiceClones: 5
    },
    silver: {
        name: 'Silver',
        basePrice: 59.99,
        features: [
            'Up to 5-minute songs',
            '10 songs per month',
            'World-class mastering',
            'Premium effects',
            '10 voice clones'
        ],
        maxDuration: 300,
        songsPerMonth: 10,
        voiceClones: 10
    },
    gold: {
        name: 'Gold',
        basePrice: 89.99,
        features: [
            'Up to 5-minute songs',
            '15 songs per month',
            'World-class mastering',
            'Premium effects',
            '15 voice clones'
        ],
        maxDuration: 300,
        songsPerMonth: 15,
        voiceClones: 15
    },
    platinum: {
        name: 'Platinum',
        basePrice: 119.99,
        features: [
            'Up to 5-minute songs',
            '20 songs per month',
            'World-class mastering',
            'Premium effects',
            '20 voice clones'
        ],
        maxDuration: 300,
        songsPerMonth: 20,
        voiceClones: 20
    },
    doublePlatinum: {
        name: 'Double Platinum',
        basePrice: 299.99,
        features: [
            'Up to 5-minute songs',
            'Unlimited songs per month',
            'World-class mastering',
            'Premium effects',
            '25 voice clones'
        ],
        maxDuration: 300,
        songsPerMonth: Infinity,
        voiceClones: 25
    }
};

// Tax rate
const TAX_RATE = 0.0825; // 8.25%

// Calculate total with tax
function calculateTotal(basePrice) {
    const tax = basePrice * TAX_RATE;
    return {
        basePrice: basePrice,
        tax: tax,
        total: basePrice + tax
    };
}

// Handle subscription button click
function handleSubscription(packageType) {
    const pkg = packages[packageType];
    const pricing = calculateTotal(pkg.basePrice);
    
    // Here you would implement the actual subscription logic
    // For example, redirecting to a payment processor or showing a payment form
    console.log(`Subscribing to ${pkg.name} package`);
    console.log(`Base Price: $${pricing.basePrice.toFixed(2)}`);
    console.log(`Tax: $${pricing.tax.toFixed(2)}`);
    console.log(`Total: $${pricing.total.toFixed(2)}`);
    console.log(`Features:`);
    pkg.features.forEach(feature => console.log(`- ${feature}`));
}

// Check if user has reached their monthly song limit
function checkSongLimit(packageType, currentSongCount) {
    const pkg = packages[packageType];
    if (pkg.songsPerMonth === Infinity) return true;
    return currentSongCount < pkg.songsPerMonth;
}

// Check if user has reached their voice clone limit
function checkVoiceCloneLimit(packageType, currentCloneCount) {
    const pkg = packages[packageType];
    return currentCloneCount < pkg.voiceClones;
}

// Initialize subscription buttons
document.addEventListener('DOMContentLoaded', () => {
    const buttons = document.querySelectorAll('.subscribe-btn');
    buttons.forEach(button => {
        button.addEventListener('click', (e) => {
            const packageType = e.target.getAttribute('data-package');
            handleSubscription(packageType);
        });
    });
});
