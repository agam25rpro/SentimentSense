// Initialize particles.js
particlesJS('particles-js', {
    particles: {
        number: { value: 80, density: { enable: true, value_area: 800 } },
        color: { value: '#8000ff' },
        shape: { type: 'circle' },
        opacity: {
            value: 0.5,
            random: true,
            animation: { enable: true, speed: 1, minimumValue: 0.1, sync: false }
        },
        size: {
            value: 3,
            random: true,
            animation: { enable: true, speed: 4, minimumValue: 0.3, sync: false }
        },
        line_linked: {
            enable: true,
            distance: 150,
            color: '#8000ff',
            opacity: 0.4,
            width: 1
        },
        move: {
            enable: true,
            speed: 2,
            direction: 'none',
            random: false,
            straight: false,
            out_mode: 'out',
            bounce: false,
        }
    },
    interactivity: {
        detect_on: 'canvas',
        events: {
            onhover: { enable: true, mode: 'repulse' },
            onclick: { enable: true, mode: 'push' },
            resize: true
        },
        modes: {
            repulse: { distance: 100, duration: 0.4 },
            push: { particles_nb: 4 }
        }
    },
    retina_detect: true
});

async function analyzeSentiment() {
    const text = document.getElementById('text-input').value.trim();
    if (!text) return;

    const API_URL = 'https://agamrampal-fastapi.hf.space/api/analyze';
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result').style.display = 'none';

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        document.getElementById('loading').style.display = 'none';
        document.getElementById('result').style.display = 'block';

        // Update sentiment badge
        const badge = document.getElementById('sentiment-badge');
        badge.textContent = `Sentiment: ${data.sentiment}`;
        badge.className = `sentiment-badge ${data.sentiment.toLowerCase()}`;

        // Create and update the graph
        const graphData = [{
            x: Object.keys(data.probabilities),
            y: Object.values(data.probabilities),
            type: 'bar',
            marker: {
                color: ['rgba(255, 99, 99, 0.7)', 'rgba(102, 179, 255, 0.7)', 'rgba(153, 255, 153, 0.7)'],
                line: {
                    color: ['#ff6363', '#66b3ff', '#99ff99'],
                    width: 2
                }
            }
        }];

        const layout = {
            title: {
                text: 'Sentiment Analysis Results',
                font: {
                    size: 24,
                    color: '#a384ff'
                }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
                family: 'Segoe UI, sans-serif',
                color: '#fff'
            },
            yaxis: {
                title: 'Probability',
                gridcolor: 'rgba(128, 0, 255, 0.1)',
                zerolinecolor: 'rgba(128, 0, 255, 0.2)',
                tickformat: ',.1%'
            },
            xaxis: {
                title: 'Sentiment',
                gridcolor: 'rgba(128, 0, 255, 0.1)',
            }
        };

        Plotly.newPlot('plotly-chart', graphData, layout);

    } catch (error) {
        console.error('Error:', error);
        document.getElementById('loading').style.display = 'none';
        alert('An error occurred while analyzing the sentiment. Please try again.');
    }
}