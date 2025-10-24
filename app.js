// CalorieLens - AI-Powered Food Calorie Tracker
// Main application logic

class CalorieTracker {
    constructor() {
        this.currentImage = null;
        this.currentAnalysis = null;
        this.meals = this.loadMeals();
        this.apiKey = localStorage.getItem('apiKey') || '';
        this.apiProvider = localStorage.getItem('apiProvider') || 'anthropic';

        this.initializeElements();
        this.attachEventListeners();
        this.updateDailySummary();
        this.renderMealHistory();
        this.loadApiConfig();
    }

    initializeElements() {
        // Input elements
        this.imageInput = document.getElementById('imageInput');
        this.apiKeyInput = document.getElementById('apiKey');
        this.apiProviderSelect = document.getElementById('apiProvider');

        // Display elements
        this.previewSection = document.getElementById('preview');
        this.previewImage = document.getElementById('previewImage');
        this.loadingSection = document.getElementById('loading');
        this.resultsSection = document.getElementById('results');
        this.uploadArea = document.getElementById('uploadArea');

        // Buttons
        this.cancelBtn = document.getElementById('cancelBtn');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.saveBtn = document.getElementById('saveBtn');
        this.saveApiKeyBtn = document.getElementById('saveApiKeyBtn');
        this.clearHistoryBtn = document.getElementById('clearHistoryBtn');

        // Summary elements
        this.dailyTotalEl = document.getElementById('dailyTotal');
        this.mealCountEl = document.getElementById('mealCount');
        this.mealHistoryEl = document.getElementById('mealHistory');
    }

    attachEventListeners() {
        this.imageInput.addEventListener('change', (e) => this.handleImageSelect(e));
        this.cancelBtn.addEventListener('click', () => this.resetUpload());
        this.analyzeBtn.addEventListener('click', () => this.analyzeFoodImage());
        this.saveBtn.addEventListener('click', () => this.saveMeal());
        this.saveApiKeyBtn.addEventListener('click', () => this.saveApiConfig());
        this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
    }

    handleImageSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            this.currentImage = e.target.result;
            this.previewImage.src = this.currentImage;
            this.uploadArea.classList.add('hidden');
            this.previewSection.classList.remove('hidden');
            this.resultsSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    resetUpload() {
        this.currentImage = null;
        this.currentAnalysis = null;
        this.imageInput.value = '';
        this.previewSection.classList.add('hidden');
        this.resultsSection.classList.add('hidden');
        this.loadingSection.classList.add('hidden');
        this.uploadArea.classList.remove('hidden');
    }

    async analyzeFoodImage() {
        if (!this.apiKey) {
            alert('Please configure your API key in the settings section below!');
            return;
        }

        this.previewSection.classList.add('hidden');
        this.loadingSection.classList.remove('hidden');

        try {
            const analysis = await this.callFoodAnalysisAPI(this.currentImage);
            this.currentAnalysis = analysis;
            this.displayResults(analysis);
        } catch (error) {
            alert('Error analyzing food: ' + error.message);
            this.resetUpload();
        }
    }

    async callFoodAnalysisAPI(imageBase64) {
        const imageData = imageBase64.split(',')[1];

        if (this.apiProvider === 'anthropic') {
            return await this.analyzeWithAnthropic(imageData);
        } else {
            return await this.analyzeWithOpenAI(imageData);
        }
    }

    async analyzeWithAnthropic(imageData) {
        const response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': this.apiKey,
                'anthropic-version': '2023-06-01'
            },
            body: JSON.stringify({
                model: 'claude-3-5-sonnet-20241022',
                max_tokens: 1024,
                messages: [{
                    role: 'user',
                    content: [
                        {
                            type: 'image',
                            source: {
                                type: 'base64',
                                media_type: 'image/jpeg',
                                data: imageData
                            }
                        },
                        {
                            type: 'text',
                            text: `Analyze this food image and provide a detailed nutritional breakdown.

Please respond in the following JSON format:
{
    "foodName": "Name of the food/dish",
    "calories": estimated total calories (number only),
    "protein": grams of protein,
    "carbs": grams of carbohydrates,
    "fat": grams of fat,
    "servingSize": estimated serving size,
    "confidence": your confidence level (low/medium/high),
    "notes": any additional observations
}

Be as accurate as possible with calorie and nutrient estimates.`
                        }
                    ]
                }]
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error?.message || 'API request failed');
        }

        const data = await response.json();
        const textContent = data.content[0].text;

        // Extract JSON from the response
        const jsonMatch = textContent.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[0]);
        }

        throw new Error('Could not parse food analysis');
    }

    async analyzeWithOpenAI(imageData) {
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            },
            body: JSON.stringify({
                model: 'gpt-4o',
                messages: [{
                    role: 'user',
                    content: [
                        {
                            type: 'image_url',
                            image_url: {
                                url: `data:image/jpeg;base64,${imageData}`
                            }
                        },
                        {
                            type: 'text',
                            text: `Analyze this food image and provide a detailed nutritional breakdown.

Please respond in the following JSON format:
{
    "foodName": "Name of the food/dish",
    "calories": estimated total calories (number only),
    "protein": grams of protein,
    "carbs": grams of carbohydrates,
    "fat": grams of fat,
    "servingSize": estimated serving size,
    "confidence": your confidence level (low/medium/high),
    "notes": any additional observations
}

Be as accurate as possible with calorie and nutrient estimates.`
                        }
                    ]
                }],
                max_tokens: 1000
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error?.message || 'API request failed');
        }

        const data = await response.json();
        const textContent = data.choices[0].message.content;

        // Extract JSON from the response
        const jsonMatch = textContent.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[0]);
        }

        throw new Error('Could not parse food analysis');
    }

    displayResults(analysis) {
        this.loadingSection.classList.add('hidden');
        this.resultsSection.classList.remove('hidden');

        document.getElementById('foodName').textContent = analysis.foodName;
        document.getElementById('calorieEstimate').textContent = `${analysis.calories} calories`;

        const nutritionHTML = `
            <p><strong>Serving Size:</strong> ${analysis.servingSize}</p>
            <p><strong>Protein:</strong> ${analysis.protein}g</p>
            <p><strong>Carbohydrates:</strong> ${analysis.carbs}g</p>
            <p><strong>Fat:</strong> ${analysis.fat}g</p>
            <p><strong>Confidence:</strong> ${analysis.confidence}</p>
            ${analysis.notes ? `<p><strong>Notes:</strong> ${analysis.notes}</p>` : ''}
        `;

        document.getElementById('nutritionDetails').innerHTML = nutritionHTML;
    }

    saveMeal() {
        if (!this.currentAnalysis) return;

        const meal = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            image: this.currentImage,
            ...this.currentAnalysis
        };

        this.meals.push(meal);
        this.saveMeals();
        this.updateDailySummary();
        this.renderMealHistory();
        this.resetUpload();

        alert('Meal saved successfully!');
    }

    deleteMeal(mealId) {
        if (!confirm('Are you sure you want to delete this meal?')) return;

        this.meals = this.meals.filter(meal => meal.id !== mealId);
        this.saveMeals();
        this.updateDailySummary();
        this.renderMealHistory();
    }

    clearHistory() {
        if (!confirm('Are you sure you want to clear all meals from today?')) return;

        this.meals = [];
        this.saveMeals();
        this.updateDailySummary();
        this.renderMealHistory();
    }

    updateDailySummary() {
        const today = new Date().toDateString();
        const todaysMeals = this.meals.filter(meal =>
            new Date(meal.timestamp).toDateString() === today
        );

        const totalCalories = todaysMeals.reduce((sum, meal) => sum + meal.calories, 0);

        this.dailyTotalEl.textContent = totalCalories;
        this.mealCountEl.textContent = todaysMeals.length;
    }

    renderMealHistory() {
        const today = new Date().toDateString();
        const todaysMeals = this.meals.filter(meal =>
            new Date(meal.timestamp).toDateString() === today
        );

        if (todaysMeals.length === 0) {
            this.mealHistoryEl.innerHTML = '<p class="empty-state">No meals logged yet. Start by adding a photo!</p>';
            return;
        }

        const mealsHTML = todaysMeals.map(meal => {
            const time = new Date(meal.timestamp).toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit'
            });

            return `
                <div class="meal-item">
                    <img src="${meal.image}" alt="${meal.foodName}" class="meal-thumbnail">
                    <div class="meal-info">
                        <h4>${meal.foodName}</h4>
                        <div class="calories">${meal.calories} cal</div>
                        <div class="time">${time}</div>
                    </div>
                    <div class="meal-actions">
                        <button onclick="tracker.deleteMeal(${meal.id})">Delete</button>
                    </div>
                </div>
            `;
        }).join('');

        this.mealHistoryEl.innerHTML = mealsHTML;
    }

    loadMeals() {
        const stored = localStorage.getItem('meals');
        return stored ? JSON.parse(stored) : [];
    }

    saveMeals() {
        localStorage.setItem('meals', JSON.stringify(this.meals));
    }

    saveApiConfig() {
        this.apiKey = this.apiKeyInput.value.trim();
        this.apiProvider = this.apiProviderSelect.value;

        if (!this.apiKey) {
            alert('Please enter an API key');
            return;
        }

        localStorage.setItem('apiKey', this.apiKey);
        localStorage.setItem('apiProvider', this.apiProvider);

        alert('API configuration saved successfully!');
    }

    loadApiConfig() {
        if (this.apiKey) {
            this.apiKeyInput.value = this.apiKey;
        }
        this.apiProviderSelect.value = this.apiProvider;
    }
}

// Initialize the app when the page loads
let tracker;
document.addEventListener('DOMContentLoaded', () => {
    tracker = new CalorieTracker();
});
