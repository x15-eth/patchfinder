"""
User Interface for Instrument & Patch Identifier

This module provides both CLI and web interfaces for the instrument identification system.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Flask imports (optional)
try:
    from flask import Flask, request, render_template, jsonify, send_file
    from werkzeug.utils import secure_filename
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Web interface disabled.")

from inference import InstrumentIdentifier


class CLI:
    """Command-line interface for instrument identification."""
    
    def __init__(self):
        self.identifier = InstrumentIdentifier()
    
    def run(self, args):
        """Run the CLI with given arguments."""
        try:
            print(f"Analyzing: {args.input}")
            
            # Run identification
            results = self.identifier.identify_instruments(
                args.input,
                use_separation=not args.no_separation,
                separation_model=args.model
            )
            
            # Save results if requested
            if args.output:
                self.identifier.save_results(results, args.output)
            
            # Display results
            self.display_results(results, args.verbose)
            
            return 0
            
        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def display_results(self, results: Dict[str, Any], verbose: bool = False):
        """Display results in a formatted way."""
        summary = results["summary"]
        
        print(f"\n{'='*50}")
        print(f"INSTRUMENT IDENTIFICATION RESULTS")
        print(f"{'='*50}")
        
        print(f"Input file: {results['input_file']}")
        print(f"Source separation: {'Yes' if results['use_separation'] else 'No'}")
        if results['use_separation']:
            print(f"Separation model: {results['separation_model']}")
        
        print(f"Overall confidence: {summary['overall_confidence']:.3f}")
        print(f"Stems analyzed: {summary['successful_analyses']}/{summary['total_stems']}")
        
        # Top instruments
        print(f"\nTOP INSTRUMENTS DETECTED:")
        print(f"{'-'*40}")
        for i, match in enumerate(summary["top_instruments"], 1):
            print(f"{i:2d}. {match['instrument']:<25} ({match['stem']:<10}) "
                  f"confidence: {match['confidence']:.3f}")
        
        # Stem-by-stem results
        if verbose or not results['use_separation']:
            print(f"\nDETAILED RESULTS BY STEM:")
            print(f"{'-'*40}")
            
            for stem_name, stem_result in results["stems"].items():
                if "error" in stem_result:
                    print(f"\n{stem_name.upper()}: FAILED")
                    print(f"  Error: {stem_result['error']}")
                else:
                    print(f"\n{stem_name.upper()}:")
                    print(f"  Duration: {stem_result['duration']:.2f}s")
                    print(f"  Confidence: {stem_result['confidence_score']:.3f}")
                    print(f"  Top matches:")
                    
                    for match in stem_result["top_matches"][:3]:
                        print(f"    {match['rank']}. {match['label']:<25} "
                              f"confidence: {match['confidence']:.3f} "
                              f"({match['occurrences']} chunks)")
        
        print(f"\n{'='*50}")


class WebInterface:
    """Flask web interface for instrument identification."""
    
    def __init__(self):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for web interface")
        
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
        self.identifier = InstrumentIdentifier()
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze():
            try:
                # Check if file was uploaded
                if 'audio_file' not in request.files:
                    return jsonify({'error': 'No audio file provided'}), 400
                
                file = request.files['audio_file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Get options
                use_separation = request.form.get('use_separation', 'true').lower() == 'true'
                separation_model = request.form.get('separation_model', '4stems')
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                upload_path = Path('temp') / filename
                upload_path.parent.mkdir(exist_ok=True)
                file.save(upload_path)
                
                # Run analysis
                results = self.identifier.identify_instruments(
                    str(upload_path),
                    use_separation=use_separation,
                    separation_model=separation_model
                )
                
                # Clean up uploaded file
                upload_path.unlink()
                
                return jsonify(results)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health')
        def health():
            return jsonify({'status': 'healthy'})
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the Flask web server."""
        print(f"Starting web interface at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_html_template():
    """Create a simple HTML template for the web interface."""
    template_dir = Path("templates")
    template_dir.mkdir(exist_ok=True)
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Instrument & Patch Identifier</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .results { margin-top: 20px; padding: 15px; background: white; border-radius: 5px; }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Instrument & Patch Identifier</h1>
        <p>Upload an audio file to identify instruments and synthesizer patches using AI.</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="audio_file">Audio File:</label>
                <input type="file" id="audio_file" name="audio_file" accept="audio/*" required>
            </div>
            
            <div class="form-group">
                <label for="use_separation">
                    <input type="checkbox" id="use_separation" name="use_separation" checked>
                    Use source separation (splits into vocals, drums, bass, other)
                </label>
            </div>
            
            <div class="form-group">
                <label for="separation_model">Separation Model:</label>
                <select id="separation_model" name="separation_model">
                    <option value="2stems">2 stems (vocals, accompaniment)</option>
                    <option value="4stems" selected>4 stems (vocals, drums, bass, other)</option>
                    <option value="5stems">5 stems (vocals, drums, bass, piano, other)</option>
                </select>
            </div>
            
            <button type="submit">Analyze Audio</button>
        </form>
        
        <div class="loading" id="loading">
            <p>🎵 Analyzing audio... This may take a few minutes.</p>
        </div>
        
        <div id="results" class="results" style="display: none;">
            <h2>Results</h2>
            <div id="resultsContent"></div>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const resultsContent = document.getElementById('resultsContent');
            
            loading.style.display = 'block';
            results.style.display = 'none';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                } else {
                    resultsContent.innerHTML = `<p class="error">Error: ${data.error}</p>`;
                }
                
                results.style.display = 'block';
            } catch (error) {
                resultsContent.innerHTML = `<p class="error">Error: ${error.message}</p>`;
                results.style.display = 'block';
            } finally {
                loading.style.display = 'none';
            }
        });
        
        function displayResults(data) {
            const summary = data.summary;
            let html = `
                <h3>Summary</h3>
                <p><strong>Overall Confidence:</strong> ${summary.overall_confidence.toFixed(3)}</p>
                <p><strong>Stems Analyzed:</strong> ${summary.successful_analyses}/${summary.total_stems}</p>
                
                <h3>Top Instruments Detected</h3>
                <ol>
            `;
            
            summary.top_instruments.forEach(match => {
                html += `<li>${match.instrument} (${match.stem}) - confidence: ${match.confidence.toFixed(3)}</li>`;
            });
            
            html += '</ol>';
            
            if (data.use_separation) {
                html += '<h3>Results by Stem</h3>';
                for (const [stemName, stemResult] of Object.entries(data.stems)) {
                    if (stemResult.error) {
                        html += `<h4>${stemName}: Failed</h4><p class="error">${stemResult.error}</p>`;
                    } else {
                        html += `
                            <h4>${stemName}</h4>
                            <p><strong>Duration:</strong> ${stemResult.duration.toFixed(2)}s</p>
                            <p><strong>Confidence:</strong> ${stemResult.confidence_score.toFixed(3)}</p>
                            <p><strong>Top matches:</strong></p>
                            <ul>
                        `;
                        stemResult.top_matches.slice(0, 3).forEach(match => {
                            html += `<li>${match.label} - confidence: ${match.confidence.toFixed(3)}</li>`;
                        });
                        html += '</ul>';
                    }
                }
            }
            
            document.getElementById('resultsContent').innerHTML = html;
        }
    </script>
</body>
</html>"""
    
    with open(template_dir / "index.html", "w") as f:
        f.write(html_content)
    
    print("Created HTML template at templates/index.html")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Instrument & Patch Identifier")
    subparsers = parser.add_subparsers(dest='mode', help='Interface mode')
    
    # CLI mode
    cli_parser = subparsers.add_parser('cli', help='Command-line interface')
    cli_parser.add_argument('input', help='Input audio file')
    cli_parser.add_argument('--no-separation', action='store_true',
                           help='Skip source separation')
    cli_parser.add_argument('--model', choices=['2stems', '4stems', '5stems'],
                           default='4stems', help='Separation model')
    cli_parser.add_argument('--output', help='Output JSON file for results')
    cli_parser.add_argument('--verbose', action='store_true',
                           help='Verbose output')
    
    # Web mode
    web_parser = subparsers.add_parser('web', help='Web interface')
    web_parser.add_argument('--host', default='127.0.0.1', help='Host address')
    web_parser.add_argument('--port', type=int, default=5000, help='Port number')
    web_parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    # Setup mode
    setup_parser = subparsers.add_parser('setup', help='Setup web templates')
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        cli = CLI()
        return cli.run(args)
    
    elif args.mode == 'web':
        if not FLASK_AVAILABLE:
            print("Error: Flask is required for web interface")
            print("Install with: pip install flask")
            return 1
        
        create_html_template()
        web = WebInterface()
        web.run(args.host, args.port, args.debug)
        return 0
    
    elif args.mode == 'setup':
        create_html_template()
        print("Web templates created successfully")
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
