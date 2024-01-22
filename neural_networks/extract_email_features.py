import re

def extract_features(email_text, keyword='free', spam_domains=['spamdomain.com']):
    # Convert to lower case for consistent processing
    email_text_lower = email_text.lower()

    # Feature 1: Frequency of a Specific Keyword
    keyword_frequency = email_text_lower.count(keyword) / len(email_text_lower.split())

    # Feature 2: Total Number of Words
    word_count = len(email_text.split())

    # Feature 3: Frequency of Exclamation Marks
    exclamation_frequency = email_text.count('!') / len(email_text)

    # Feature 4: Presence of a Spam-Associated Domain
    spam_domain_present = any(domain in email_text_lower for domain in spam_domains)

    # Feature 5: Ratio of Uppercase to Lowercase Letters
    uppercase_count = sum(1 for c in email_text if c.isupper())
    lowercase_count = sum(1 for c in email_text if c.islower())
    uppercase_ratio = uppercase_count / (lowercase_count + 1)  # +1 to avoid division by zero

    return [keyword_frequency, word_count, exclamation_frequency, int(spam_domain_present), uppercase_ratio]

# Example usage
email_text = "Get your FREE trial now! Visit spamdomain.com for more details."
features = extract_features(email_text)
print(features)
