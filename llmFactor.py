from numpy.f2py.auxfuncs import throw_error
from openai import OpenAI
import pandas as pd
import datetime
from dataset_manager import NewsDataLoader, PriceDataLoader
from typing import List, Dict, Any, Tuple
import argparse
import json
from pathlib import Path
import sys
from tqdm import tqdm

class LLMFactorAnalyzer:
    def __init__(self, base_url: str, api_key: str, model: str):
        """
        Initialize the LLM Factor Analyzer.
        
        Args:
            base_url: Base URL for the OpenAI API
            api_key: API key for authentication
            model: Model identifier to use for analysis
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self.price_data = PriceDataLoader()
        self.news_data = NewsDataLoader()
        
    def get_available_tickers(self) -> List[str]:
        """Get tickers available in both price and news data."""
        price_tickers = set(self.price_data.get_available_tickers())
        news_tickers = set(self.news_data.get_available_tickers())
        return sorted(tuple(price_tickers.intersection(news_tickers)))
    
    def get_available_dates(self, ticker: str, price_k: int) -> List[datetime.datetime]:
        """Get dates available in both price and news data for a given ticker."""
        price_dates = set(self.price_data.get_available_dates(ticker, price_k))
        news_dates = set(self.news_data.get_available_dates(ticker))
        return sorted(price_dates.intersection(news_dates))
    
    def format_price_movements(self, 
                             price_movements: List[Dict[str, Any]], 
                             stock_target: str,
                             target_date: datetime.datetime) -> str:
        """Format price movement data into a string."""
        price_str_format = "On {date}, the stock price of {stock_target} {risefall}.\n"
        price_str = ""
        
        for move in price_movements[:-1]:
            price_str += price_str_format.format(
                date=move['date'].strftime('%Y-%m-%d'),
                stock_target=stock_target,
                risefall="rose" if move['rise'] else "fell"
            )
            
        price_str_last = price_str_format.format(
            date=target_date.strftime('%Y-%m-%d'),
            stock_target=stock_target,
            risefall="____"
        )
        
        return price_str, price_str_last
    
    def analyze_factors(self, 
                       ticker: str,
                       target_date: datetime.datetime,
                       factor_k: int = 5,
                       price_k: int = 5) -> Dict[str, Any]:
        """
        Analyze factors affecting stock price movement.
        
        Args:
            ticker: Stock ticker symbol
            target_date: Target date for analysis
            factor_k: Number of factors to extract
            price_k: Number of price movement days to consider
            
        Returns:
            Dictionary containing analysis results and metadata
        """

        result = {
            "ticker": ticker,
            "date": target_date.strftime('%Y-%m-%d'),
            "factors": None,
            "analysis": None,
            "prediction": None,
            "actual": None,
            "status": "",
            "error": ""
        }

        try:
            # Get data
            news_str = self.news_data.get_news_by_date(ticker, target_date)
            price_movements = self.price_data.get_price_movements(ticker, target_date, price_k)
            price_str, price_str_last = self.format_price_movements(price_movements, ticker, target_date)
            
            # Extract factors
            factor_extraction = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": f"Please extract the top {factor_k} factors that may affect the stock price of {ticker} from the following news."},
                    {"role": "user", "content": news_str}
                ]
            )
            factor_str = factor_extraction.choices[0].message.content
            result['factors'] = factor_str
            
            # Analyze price movement
            answer_extraction = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system",
                     "content": "Based on the following information, please judge the direction of the stock price from rise/fall, fill in the blank and give reasons."},
                    {"role": "system",
                     "content": f"These are the main factors that may affect this stock’s price recently: {factor_str}."},
                    {"role": "system", "content": price_str},
                    {"role": "assistant", "content": price_str_last},
                ]
            )

            answer = answer_extraction.choices[0].message.content
            result['analysis'] = answer

            # Parse prediction
            filled_blanks = answer.split('\n')[0]
            positive_sentiments = ["rise", "rose"]
            negative_sentiments = ["fall", "fell"]

            pred_rise = any(sentiment in filled_blanks for sentiment in positive_sentiments)
            pred_fall = any(sentiment in filled_blanks for sentiment in negative_sentiments)
            actual_rise = price_movements[-1]['rise']

            if pred_rise == pred_fall:
                raise ValueError("Prediction is uncertain.")

            result['prediction'] = "rise" if pred_rise else "fall"
            result['actual'] = "rise" if actual_rise else "fall"

            result['status'] = "success"
            return result
            
        except Exception as e:
            result['status'] = "error"
            result['error'] = str(e)
            return result

def parse_args():
    parser = argparse.ArgumentParser(description='LLM Factor Analysis Tool')
    parser.add_argument('--endpoint', type=str, default='http://localhost:8000/v1',
                      help='API endpoint URL (default: http://localhost:8000/v1)')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                      help='Model identifier to use (default: meta-llama/Llama-3.2-3B-Instruct)')
    parser.add_argument('--token', type=str, default='token-abc123',
                      help='API token (default: token-abc123)')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results (default: results)')
    parser.add_argument('--tickers', type=str, nargs='+',
                      help='Specific tickers to analyze (optional, analyzes all available if not specified)')
    parser.add_argument('--start-date', type=str,
                      help='Start date for analysis (YYYY-MM-DD format, optional)')
    parser.add_argument('--end-date', type=str,
                      help='End date for analysis (YYYY-MM-DD format, optional)')
    return parser.parse_args()

def main():
    try:
        # Parse command line arguments
        args = parse_args()

        # Initialize analyzer
        analyzer = LLMFactorAnalyzer(args.endpoint, args.token, args.model)

        # Get tickers to analyze
        available_tickers = analyzer.get_available_tickers()
        if not available_tickers:
            print("No tickers available in both price and news data.")
            return

        tickers_to_analyze = args.tickers if args.tickers else available_tickers
        if not set(tickers_to_analyze).issubset(available_tickers):
            print("Invalid tickers specified.")
            print(f"Please remove the following invalid tickers: {set(tickers_to_analyze) - set(available_tickers)}")
            return

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each ticker
        stat = {
            "total": 0,
            "success": 0,
            "uncertain": 0,
            "error": 0,
            "confusion_rate": {
                "true_rise": 0,
                "true_fall": 0,
                "false_rise": 0,
                "false_fall": 0,
            }
        }
        all_results = []
        for ticker in tqdm(tickers_to_analyze, desc="Processing tickers"):
            # Get available dates for this ticker
            dates = analyzer.get_available_dates(ticker, price_k=5)
            if not dates:
                print(f"No dates available for ticker {ticker}")
                continue

            # Filter dates if specified
            if args.start_date:
                start_date = pd.to_datetime(args.start_date)
                dates = [d for d in dates if d >= start_date]
            if args.end_date:
                end_date = pd.to_datetime(args.end_date)
                dates = [d for d in dates if d <= end_date]

            # Analyze each date
            for target_date in tqdm(dates, desc=f"Analyzing {ticker}", leave=False):
                result = analyzer.analyze_factors(ticker, target_date)
                all_results.append(result)

                # Update stats
                stat['total'] += 1
                if result['status'] == 'success':
                    stat['success'] += 1
                    if result['prediction'] == result['actual']:
                        if result['prediction'] == 'rise':
                            stat['confusion_rate']['true_rise'] += 1
                        else:
                            stat['confusion_rate']['true_fall'] += 1
                    else:
                        if result['prediction'] == 'rise':
                            stat['confusion_rate']['false_rise'] += 1
                        else:
                            stat['confusion_rate']['false_fall'] += 1
                elif result['status'] == 'error':
                    stat['error'] += 1
                elif result['status'] == 'uncertain':
                    stat['uncertain'] += 1

                # Save combined results
                combined_output = output_dir / "all_results.json"
                with open(combined_output, 'w') as f:
                    json.dump(all_results, f, indent=2)

                stat_output = output_dir / "summary.json"
                with open(stat_output, 'w') as f:
                    json.dump(stat, f, indent=2)

        print(f"\nAnalysis complete. Results saved to {output_dir}")

        # Print summary
        successful = sum(1 for r in all_results if r['status'] == 'success')
        failed = sum(1 for r in all_results if r['status'] == 'error')

        # Print summary
        print(f"\nSummary:")
        print(f"Total analyses: {len(all_results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Uncertain: {stat['uncertain']}")

        # Confusion rate
        print(f"Confusion rate:")
        print(f"  True rise: {stat['confusion_rate']['true_rise'] / successful:.2f}")
        print(f"  True fall: {stat['confusion_rate']['true_fall'] / successful:.2f}")
        print(f"  False rise: {stat['confusion_rate']['false_rise'] / successful:.2f}")
        print(f"  False fall: {stat['confusion_rate']['false_fall'] / successful:.2f}")

        # Benchmark (ACC, F-1, MCC)
        tr = stat['confusion_rate']['true_rise']
        tf = stat['confusion_rate']['true_fall']
        fr = stat['confusion_rate']['false_rise']
        ff = stat['confusion_rate']['false_fall']

        acc = (tr + tf) / successful

        f1_denominator = 2 * tr + fr + ff
        f1 = 2 * tr / f1_denominator if f1_denominator != 0 else float('nan')

        mcc_denominator = ((tr + fr) * (tr + ff) * (tf + fr) * (tf + ff)) ** 0.5
        mcc = (tr * tf - fr * ff) / mcc_denominator if mcc_denominator != 0 else float('nan')

        print(f"\nBenchmark:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Matthews Correlation Coefficient: {mcc:.4f}")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted.")
        return

    except Exception as e:
        print(f"Unexpected Error: {e}")
        raise e

if __name__ == "__main__":
    main()