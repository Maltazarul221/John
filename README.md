# CalorieLens - AI-Powered Food Calorie Tracker

Track your daily calorie intake by simply taking pictures of your food! CalorieLens uses advanced AI vision models to analyze your meals and provide detailed nutritional information.

## Features

- **AI Food Recognition** - Uses Claude or GPT-4 Vision to identify food and estimate calories
- **Photo Capture** - Take photos directly from your device camera or upload existing images
- **Daily Tracking** - Automatic daily calorie totals and meal counts
- **Meal History** - View all meals logged today with thumbnails
- **Nutrition Details** - Get breakdown of calories, protein, carbs, and fat
- **Local Storage** - All data stored locally in your browser for privacy
- **Responsive Design** - Works on desktop and mobile devices

## Screenshots

The app includes:
- Daily summary dashboard with total calories and meal count
- Easy photo upload with camera access
- Real-time AI food analysis
- Detailed nutrition breakdown
- Meal history with thumbnails and timestamps

## Getting Started

### Prerequisites

You'll need an API key from one of these providers:
- **Anthropic Claude** (recommended) - Get your key at https://console.anthropic.com/
- **OpenAI GPT-4 Vision** - Get your key at https://platform.openai.com/

### Installation

1. Clone or download this repository
2. Open `index.html` in a web browser
3. Configure your API key in the settings section at the bottom

### Usage

1. **Configure API Key**
   - Scroll to the "API Configuration" section
   - Select your AI provider (Anthropic Claude or OpenAI)
   - Enter your API key
   - Click "Save Key"

2. **Add a Meal**
   - Click "Take/Upload Photo"
   - Take a photo or select an existing image
   - Click "Analyze Food"
   - Review the nutritional information
   - Click "Save to Today's Log"

3. **Track Your Progress**
   - View your daily calorie total at the top
   - See all meals logged today in the history section
   - Delete individual meals or clear all history

## How It Works

### AI Analysis

The app sends your food image to an AI vision model with a specific prompt asking for:
- Food identification
- Calorie estimation
- Macronutrient breakdown (protein, carbs, fat)
- Serving size estimation
- Confidence level

### Data Storage

All data is stored locally in your browser using:
- **localStorage** for API configuration and meal history
- **Base64 encoding** for storing meal images

No data is sent to any server except the AI API for food analysis.

## API Providers

### Anthropic Claude (Recommended)

- Model: claude-3-5-sonnet-20241022
- Great accuracy for food recognition
- Detailed nutritional analysis
- Cost: ~$0.003 per image

### OpenAI GPT-4 Vision

- Model: gpt-4o
- Excellent food recognition
- Comprehensive nutrition data
- Cost: ~$0.01 per image

## File Structure

```
calorie-tracker/
├── index.html          # Main HTML structure
├── styles.css          # Styling and responsive design
├── app.js              # Core application logic
└── README.md           # This file
```

## Technical Details

### Key Components

1. **CalorieTracker Class** - Main application controller
   - Handles image upload and preview
   - Manages API calls to AI services
   - Stores and retrieves meal data
   - Updates UI and summary statistics

2. **API Integration**
   - Supports both Anthropic and OpenAI
   - Sends base64-encoded images with structured prompts
   - Parses JSON responses for nutrition data

3. **Local Storage**
   - Persists API configuration
   - Stores meal history with images
   - Filters meals by date for daily tracking

### Browser Compatibility

- Modern browsers with ES6+ support
- Chrome, Firefox, Safari, Edge (latest versions)
- Mobile browsers with camera access support

## Privacy & Security

- All data stored locally in your browser
- API keys stored in localStorage (not sent anywhere except to the AI provider)
- No external analytics or tracking
- Images only sent to the AI provider for analysis

## Customization

### Changing the AI Prompt

Edit the prompt in `app.js` in the `analyzeWithAnthropic` or `analyzeWithOpenAI` methods to adjust:
- Specificity of nutritional data
- Additional information requested
- Format of the response

### Styling

Modify `styles.css` to customize:
- Color scheme (current: purple gradient)
- Layout and spacing
- Mobile responsiveness breakpoints

## Limitations

- Calorie estimates are approximations based on visual analysis
- Accuracy depends on image quality and food visibility
- Complex mixed dishes may have less accurate estimates
- Requires internet connection for AI analysis

## Future Enhancements

Potential features to add:
- Multi-day history and trends
- Weekly/monthly calorie charts
- Custom calorie goals and tracking
- Export data to CSV
- Barcode scanning for packaged foods
- Recipe database integration
- Multi-language support

## Troubleshooting

### "Please configure your API key" error
- Make sure you've entered your API key in the settings section
- Verify the API key is correct and active

### "API request failed" error
- Check your API key is valid
- Ensure you have sufficient credits with your AI provider
- Check your internet connection

### Images not analyzing
- Ensure the image is clear and well-lit
- Try a different angle or closer shot
- Check browser console for specific errors

### Data not persisting
- Check that your browser allows localStorage
- Clear browser cache and try again
- Try a different browser

## Credits

Built with:
- Vanilla JavaScript (ES6+)
- Anthropic Claude API
- OpenAI GPT-4 Vision API

## License

MIT License - Feel free to use and modify for your own projects!

## Contributing

Suggestions and improvements welcome! This is a simple web app that can be extended in many ways.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the browser console for error messages
3. Verify API key and provider settings

---

**Note**: This app is for educational and personal use. Calorie estimates are approximations and should not replace professional dietary advice.
