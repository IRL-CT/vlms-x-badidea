# GitHub Pages Website

This directory contains the GitHub Pages website for the "Bad Idea or Good Prediction?" project.

## Setup Instructions

To enable GitHub Pages for this repository:

1. Go to your repository settings on GitHub
2. Navigate to "Pages" in the left sidebar
3. Under "Build and deployment":
   - Source: Deploy from a branch
   - Branch: Select `main` and `/docs` folder
   - Click "Save"

The website will be available at: `https://[your-username].github.io/vlms-x-badidea/`

## Structure

```
docs/
├── index.html          # Main website page
├── static/
│   ├── css/           # Stylesheets (Bulma framework)
│   ├── js/            # JavaScript files
│   ├── images/        # Image assets
│   └── videos/        # Video files (if any)
└── README.md          # This file
```

## Customization

The website is built using the [Bulma CSS framework](https://bulma.io/) and is based on the [Nerfies template](https://github.com/nerfies/nerfies.github.io).

To customize:
- Edit `index.html` to update content
- Add new images to `static/images/`
- Modify styles in `static/css/index.css`

## Local Testing

To test the website locally, you can use Python's built-in HTTP server:

```bash
cd docs
python3 -m http.server 8000
```

Then visit `http://localhost:8000` in your browser.

## License

This website is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
