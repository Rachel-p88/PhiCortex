import re
import math
from urllib.parse import urlparse

SHORTENERS = {'bit.ly','goo.gl','tinyurl.com','ow.ly','t.co','tiny.cc',
              'is.gd','buff.ly','adf.ly','shorte.st','cutt.ly','rb.gy',
              'shorturl.at','snip.ly','bl.ink'}

SUSPICIOUS_TLDS = {'.xyz','.top','.club','.online','.site',
                   '.biz','.tk','.ml','.ga','.cf','.gq','.pw','.work'}

PHISH_WORDS = ['verify','update','secure','account','login','signin',
               'banking','confirm','password','credential','ebayisapi',
               'webscr','paypal','free','lucky','winner','prize',
               'suspend','unusual','validate','recover']

def entropy(s):
    if not s: return 0
    freq = {c: s.count(c)/len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values())

def extract_features(url):
    try:
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'http://' + url

        parsed       = urlparse(url)
        domain       = parsed.netloc.lower()
        domain_clean = re.sub(r'^www\.', '', domain.split(':')[0])
        path         = parsed.path.lower()
        query        = parsed.query.lower()
        full         = url.lower()
        parts        = domain_clean.split('.')

        features = {}

        features['using_ip']       = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain_clean) else 0
        features['url_length']     = len(url)
        features['short_url']      = 1 if any(s in domain_clean for s in SHORTENERS) else 0
        features['symbol_at']      = 1 if '@' in url else 0
        features['redirecting']    = 1 if '//' in path else 0
        features['prefix_suffix']  = 1 if '-' in domain_clean else 0
        features['sub_domains']    = max(0, len(parts) - 2)
        features['has_https']      = 1 if parsed.scheme == 'https' else 0
        features['https_token']    = 1 if 'https' in domain_clean else 0
        features['domain_length']  = len(domain_clean)
        features['nb_dots']        = full.count('.')
        features['nb_hyphens']     = full.count('-')
        features['nb_at']          = full.count('@')
        features['nb_qm']          = full.count('?')
        features['nb_eq']          = full.count('=')
        features['nb_slash']       = full.count('/')
        features['nb_digits']      = sum(c.isdigit() for c in url)
        features['ratio_digits']   = round(features['nb_digits'] / max(len(url), 1), 4)
        features['port']           = 1 if (parsed.port and parsed.port not in [80, 443]) else 0
        features['suspicious_tld'] = 1 if any(domain_clean.endswith(t) for t in SUSPICIOUS_TLDS) else 0
        features['phish_hints']    = sum(1 for w in PHISH_WORDS if w in full)
        features['path_length']    = len(path)
        features['query_length']   = len(query)
        features['nb_params']      = len(query.split('&')) if query else 0
        features['info_email']     = 1 if 'mailto:' in full else 0
        features['url_depth']      = len([p for p in path.split('/') if p])
        features['has_fragment']   = 1 if parsed.fragment else 0
        features['domain_entropy'] = round(entropy(domain_clean), 4)

        return features

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None
