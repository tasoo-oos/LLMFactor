from typing import Dict, Any


class StatisticsTracker:
    def __init__(self):
        self.stats = {
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

    def update(self, result: Dict[str, Any]) -> None:
        """Update statistics based on analysis result."""
        self.stats['total'] += 1

        if result['status'] == 'success':
            self.stats['success'] += 1
            if result['prediction'] == result['actual']:
                if result['prediction'] == 'rise':
                    self.stats['confusion_rate']['true_rise'] += 1
                else:
                    self.stats['confusion_rate']['true_fall'] += 1
            else:
                if result['prediction'] == 'rise':
                    self.stats['confusion_rate']['false_rise'] += 1
                else:
                    self.stats['confusion_rate']['false_fall'] += 1
        elif result['status'] == 'error':
            self.stats['error'] += 1
        elif result['status'] == 'uncertain':
            self.stats['uncertain'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self.stats

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate benchmark metrics."""
        successful = self.stats['success']
        if successful == 0:
            return {
                "accuracy": float('nan'),
                "f1_score": float('nan'),
                "mcc": float('nan')
            }

        tr = self.stats['confusion_rate']['true_rise']
        tf = self.stats['confusion_rate']['true_fall']
        fr = self.stats['confusion_rate']['false_rise']
        ff = self.stats['confusion_rate']['false_fall']

        acc = (tr + tf) / successful

        f1_denominator = 2 * tr + fr + ff
        f1 = 2 * tr / f1_denominator if f1_denominator != 0 else float('nan')

        mcc_denominator = ((tr + fr) * (tr + ff) * (tf + fr) * (tf + ff)) ** 0.5
        mcc = (tr * tf - fr * ff) / mcc_denominator if mcc_denominator != 0 else float('nan')

        return {
            "accuracy": acc,
            "f1_score": f1,
            "mcc": mcc
        }
