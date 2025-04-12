// Enhanced package definitions with distribution and radio features
const packages = {
    bronze: {
        name: 'Bronze',
        basePrice: 89.99,
        features: [
            '10 songs per month',
            '5 voice clones',
            'Unlimited distribution',
            'Professional mastering'
        ],
        maxDuration: 300,
        songsPerMonth: 10,
        voiceClones: 5,
        distribution: {
            platforms: ['Spotify', 'Apple Music', 'Amazon Music'],
            type: 'Basic'
        }
    },
    silver: {
        name: 'Silver',
        basePrice: 119.99,
        features: [
            '20 songs per month',
            '10 voice clones',
            'Unlimited distribution',
            'Radio submission ready',
            'Professional mastering'
        ],
        maxDuration: 300,
        songsPerMonth: 20,
        voiceClones: 10,
        distribution: {
            platforms: ['Spotify', 'Apple Music', 'Amazon Music', 'YouTube Music', 'Tidal'],
            type: 'Premium'
        },
        radioSubmission: true
    },
    gold: {
        name: 'Gold',
        basePrice: 149.99,
        features: [
            '30 songs per month',
            '15 voice clones',
            'Unlimited distribution',
            'Radio campaign',
            'Priority support',
            'Professional mastering'
        ],
        maxDuration: 300,
        songsPerMonth: 30,
        voiceClones: 15,
        distribution: {
            platforms: ['Spotify', 'Apple Music', 'Amazon Music', 'YouTube Music', 'Tidal', 'Deezer'],
            type: 'Premium'
        },
        radioSubmission: true,
        radioCampaign: true
    },
    platinum: {
        name: 'Platinum',
        basePrice: 269.99,
        features: [
            '40 songs per month',
            '20 voice clones',
            'Unlimited distribution',
            'Radio campaign',
            'Priority support',
            'Professional mastering'
        ],
        maxDuration: 300,
        songsPerMonth: 40,
        voiceClones: 20,
        distribution: {
            platforms: ['Spotify', 'Apple Music', 'Amazon Music', 'YouTube Music', 'Tidal', 'Deezer', 'Pandora'],
            type: 'Elite'
        },
        radioSubmission: true,
        radioCampaign: true,
        prioritySupport: true
    },
    doublePlatinum: {
        name: 'Double Platinum',
        basePrice: 699.99,
        features: [
            'Unlimited songs per month',
            '25 voice clones',
            'Unlimited distribution',
            'Radio campaign',
            'Label services',
            'Professional mastering'
        ],
        maxDuration: 300,
        songsPerMonth: Infinity,
        voiceClones: 25,
        distribution: {
            platforms: ['Spotify', 'Apple Music', 'Amazon Music', 'YouTube Music', 'Tidal', 'Deezer', 'Pandora', 'SoundCloud'],
            type: 'Ultimate'
        },
        radioSubmission: true,
        radioCampaign: true,
        prioritySupport: true,
        labelServices: true
    }
};

// Tax rate
const TAX_RATE = 0.0825; // 8.25%

class MusicGenerator {
    constructor(packageType) {
        this.package = packages[packageType];
        this.currentSongCount = 0;
        this.currentCloneCount = 0;
    }

    async generateMusic(songData) {
        try {
            const response = await fetch('/api/generate-music', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ...songData,
                    packageType: this.package.name
                })
            });

            if (!response.ok) {
                throw new Error('Failed to generate music');
            }

            const result = await response.json();
            return this.handleGenerationResult(result);
        } catch (error) {
            console.error('Error generating music:', error);
            throw error;
        }
    }

    handleGenerationResult(result) {
        if (result.status === 'success') {
            this.currentSongCount++;
            return {
                ...result,
                remainingSongs: this.getRemainingResources().songs,
                remainingClones: this.getRemainingResources().clones
            };
        }
        throw new Error(result.message || 'Failed to generate music');
    }

    getRemainingResources() {
        return {
            songs: this.package.songsPerMonth === Infinity ? 
                   Infinity : 
                   this.package.songsPerMonth - this.currentSongCount,
            clones: this.package.voiceClones - this.currentCloneCount
        };
    }

    async distributeMusic(songId, platforms) {
        try {
            const response = await fetch('/api/distribute-music', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    songId,
                    platforms,
                    packageType: this.package.name
                })
            });

            if (!response.ok) {
                throw new Error('Failed to distribute music');
            }

            return await response.json();
        } catch (error) {
            console.error('Error distributing music:', error);
            throw error;
        }
    }

    async submitToRadio(songId) {
        if (!this.package.radioSubmission) {
            throw new Error('Radio submission not available in current package');
        }

        try {
            const response = await fetch('/api/radio-submission', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    songId,
                    packageType: this.package.name
                })
            });

            if (!response.ok) {
                throw new Error('Failed to submit to radio');
            }

            return await response.json();
        } catch (error) {
            console.error('Error submitting to radio:', error);
            throw error;
        }
    }
}

// UI Handlers
document.addEventListener('DOMContentLoaded', () => {
    const subscribeButtons = document.querySelectorAll('button');
    
    subscribeButtons.forEach(button => {
        button.addEventListener('click', async (e) => {
            const packageCard = e.target.closest('.package-card');
            const packageType = packageCard.querySelector('h2').textContent.toLowerCase();
            
            try {
                const generator = new MusicGenerator(packageType);
                await handleSubscription(packageType, generator);
            } catch (error) {
                console.error('Subscription error:', error);
                alert('Failed to process subscription. Please try again.');
            }
        });
    });
});

async function handleSubscription(packageType, generator) {
    const pkg = packages[packageType];
    const pricing = calculateTotal(pkg.basePrice);
    
    // Show subscription confirmation modal
    showModal({
        title: `Subscribe to ${pkg.name}`,
        content: `
            <div class="text-lg mb-4">
                <p>Base Price: $${pricing.basePrice.toFixed(2)}</p>
                <p>Tax: $${pricing.tax.toFixed(2)}</p>
                <p class="font-bold">Total: $${pricing.total.toFixed(2)}/month</p>
            </div>
            <div class="mb-4">
                <h3 class="font-bold mb-2">Package Features:</h3>
                <ul class="list-disc pl-5">
                    ${pkg.features.map(feature => `<li>${feature}</li>`).join('')}
                </ul>
            </div>
            <div class="mb-4">
                <h3 class="font-bold mb-2">Distribution Platforms:</h3>
                <p>${pkg.distribution.platforms.join(', ')}</p>
            </div>
            ${pkg.radioSubmission ? `
                <div class="mb-4">
                    <h3 class="font-bold mb-2">Radio Submission:</h3>
                    <p>Professional radio submission package included</p>
                </div>
            ` : ''}
        `,
        onConfirm: async () => {
            // Process subscription
            try {
                const response = await fetch('/api/subscribe', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        packageType,
                        pricing
                    })
                });

                if (!response.ok) {
                    throw new Error('Subscription failed');
                }

                // Redirect to dashboard or show success message
                window.location.href = '/dashboard';
            } catch (error) {
                console.error('Subscription error:', error);
                alert('Failed to process subscription. Please try again.');
            }
        }
    });
}

function calculateTotal(basePrice) {
    const tax = basePrice * TAX_RATE;
    return {
        basePrice: basePrice,
        tax: tax,
        total: basePrice + tax
    };
}

function showModal({ title, content, onConfirm }) {
    // Implementation of modal UI
    // This would create and show a modal dialog with the provided content
    // and handle the confirmation action
}
