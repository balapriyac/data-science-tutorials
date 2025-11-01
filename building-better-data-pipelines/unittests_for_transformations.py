import unittest

class TestEventTransformations(unittest.TestCase):
    def test_valid_event_processing(self):
        """Test processing of valid events"""
        event = {
            'user_id': 123,
            'event_type': 'click',
            'value': 50
        }
        
        result = process_event_good(event, '2025-01-01')
        
        self.assertIn('id', result)
        self.assertEqual(result['user_id'], 123)
        self.assertEqual(result['processing_date'], '2025-01-01')
    
    def test_idempotency(self):
        """Verify same input produces same output"""
        event = {'user_id': 123, 'value': 50, 'timestamp': '2025-01-01'}
        date = '2025-01-01'
        
        result1 = process_event_good(event, date)
        result2 = process_event_good(event, date)
        
        self.assertEqual(result1, result2)
    
    def test_null_handling(self):
        """Test that nulls are handled correctly"""
        df = pd.DataFrame({
            'user_id': [1, None, 3],
            'value': [10, 20, 30]
        })
        
        with self.assertRaises(ValueError):
            validate_events(df, context="test")
    
    def test_invalid_values(self):
        """Test that invalid values are caught"""
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'value': [10, -20, 30]
        })
        
        with self.assertRaises(ValueError) as context:
            validate_events(df, context="test")
        
        self.assertIn('negative', str(context.exception).lower())

if __name__ == '__main__':
    unittest.main()
