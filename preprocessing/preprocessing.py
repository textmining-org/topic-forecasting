import pandas as pd
import treform as ptm
import pickle
import os
import re

#이걸로 정했다!

def preprocess_text(text):
    """
    Preprocess text to treat specific blockchain-related multi-word terms as single tokens.
    """
    terms_to_replace = {
        'Atomic Swap': 'Atomic-Swap',
        'Blockchain Explorer': 'Blockchain-Explorer',
        'Blockchain Platform': 'Blockchain-Platform',
        'Blockchain Technology': 'Blockchain-Technology',
        'Cold Wallet': 'Cold-Wallet',
        'Consensus Algorithm': 'Consensus-Algorithm',
        'Cross Chain': 'Cross-Chain',
        'Cryptocurrency Mining': 'Cryptocurrency-Mining',
        'Cryptographic Hash': 'Cryptographic-Hash',
        'Decentralized Application': 'Decentralized-Application',
        'Decentralized Exchange': 'Decentralized-Exchange',
        'Delegated Proof of Stake': 'Delegated-Proof-of-Stake',
        'Digital Asset': 'Digital-Asset',
        'Digital Currency': 'Digital-Currency',
        'Distributed Ledger Technology': 'Distributed-Ledger-Technology',
        'ERC-20 Token': 'ERC-20-Token',
        'ERC-721 Token': 'ERC-721-Token',
        'Gas Fees': 'Gas-Fees',
        'Hot Wallet': 'Hot-Wallet',
        'Internet of Things': 'Internet-of-Things',
        'Layer 2': 'Layer-2',
        'Lightning Network': 'Lightning-Network',
        'Multi-Signature Wallet': 'Multi-Signature-Wallet',
        'Multi Signature Wallet': 'Multi-Signature-Wallet',
        'Non-fungible Token': 'Non-fungible-Token',
        'Non fungible Token': 'Non-fungible-Token',
        'Off-Chain Transaction': 'Off-Chain-Transaction',
        'Off Chain Transaction': 'Off-Chain-Transaction',
        'On-Chain Transaction': 'On-Chain-Transaction',
        'On Chain Transaction': 'On-Chain-Transaction',
        'Private Blockchain': 'Private-Blockchain',
        'Proof of Authority': 'Proof-of-Authority',
        'Proof of Identity': 'Proof-of-Identity',
        'Public Blockchain': 'Public-Blockchain',
        'Public Ledger': 'Public-Ledger',
        'Security Token Offering': 'Security-Token-Offering',
        'Sharding': 'Sharding',
        'Side Chain': 'Side-Chain',
        'Smart Contract': 'Smart-Contract',
        'Smart Contract Audit': 'Smart-Contract-Audit',
        'Smart Contracts': 'Smart-Contracts',
        'Soft Fork': 'Soft-Fork',
        'Hard Fork': 'Hard-Fork',
        'State Channel': 'State-Channel',
        'Token Standard': 'Token-Standard',
        'Yield Farming': 'Yield-Farming',
        'Liquidity Pool': 'Liquidity-Pool',
        'Flash Loan': 'Flash-Loan',
        'Wrapped Tokens': 'Wrapped-Tokens',
        'Governance Token': 'Governance-Token',
        'Application-Specific Integrated Circuit': 'Application-Specific-Integrated-Circuit',
        'Application Specific Integrated Circuit': 'Application-Specific-Integrated-Circuit',
        'Block Depth': 'Block-Depth',
        'Block Explorer': 'Block-Explorer',
        'Block Height': 'Block-Height',
        'Block Reward': 'Block-Reward',
        'Banking Secrecy Act': 'Banking-Secrecy-Act',
        'Byzantine Fault Tolerance': 'Byzantine-Fault-Tolerance',
        'Certificate Authority': 'Certificate-Authority',
        'Closed Source': 'Closed-Source',
        'Command-Line Interface': 'Command-Line-Interface',
        'Command Line Interface': 'Command-Line-Interface',
        'Consensus Mechanism': 'Consensus-Mechanism',
        'Cryptocurrency Exchange': 'Cryptocurrency-Exchange',
        'Cryptocurrency Wallet': 'Cryptocurrency-Wallet',
        'Decentralized Autonomous Organization': 'Decentralized-Autonomous-Organization',
        'Decentralized Finance': 'Decentralized-Finance',
        'Directed Acyclic Graph': 'Directed-Acyclic-Graph',
        'Double Spend Attack': 'Double-Spend-Attack',
        'Ethereum Enterprise Alliance': 'Ethereum-Enterprise-Alliance',
        'Ethereum Virtual Machine': 'Ethereum-Virtual-Machine',
        'Financial Crimes Enforcement Network': 'Financial-Crimes-Enforcement-Network',
        'Gossip Protocol': 'Gossip-Protocol',
        'Graphical User Interface': 'Graphical-User-Interface',
        'Hash Collision': 'Hash-Collision',
        'HASH FUNCTION': 'HASH-FUNCTION',
        'HEXADECIMAL NOTATION': 'HEXADECIMAL-NOTATION',
        'Initial Coin Offering': 'Initial-Coin-Offering',
        'Initial Token Offering': 'Initial-Token-Offering',
        'Merkle Proof': 'Merkle-Proof',
        'Merkle Tree': 'Merkle-Tree',
        'Merkle Root': 'Merkle-Root',
        'Mining Pool': 'Mining-Pool',
        'CPU MINER': 'CPU-MINER',
        'GPU MINER': 'GPU-MINER',
        'ASIC MINER': 'ASIC-MINER',
        'MINING POOL': 'MINING-POOL',
        'MONEY TRANSMITTING': 'MONEY-TRANSMITTING',
        'FULL NODE': 'FULL-NODE',
        'LIGHT NODE': 'LIGHT-NODE',
        'Peer-to-Peer Network': 'Peer-to-Peer-Network',
        'Peer to Peer Network': 'Peer-to-Peer-Network',
        'Private Key': 'Private-Key',
        'Private Key Infrastructure': 'Private-Key-Infrastructure',
        'Proof of Liquidity': 'Proof-of-Liquidity',
        'Proof of Stake': 'Proof-of-Stake',
        'Proof of Work': 'Proof-of-Work',
        'Public Key': 'Public-Key',
        'STABLE COIN': 'STABLE COIN',
        'Ring Signature': 'Ring-Signature',
        'Secure Hash Algorithm': 'Secure-Hash-Algorithm',
        'SECURITIES AND EXCHANGE COMMISSION': 'SECURITIES-AND-EXCHANGE-COMMISSION',
        'SECURITY TOKEN OFFERING': 'SECURITY-TOKEN-OFFERING',
        'Simple Agreement for Future Tokens': 'Simple-Agreement-for-Future-Tokens',
        'Total Complete': 'Total-Complete',
        'STATE MACHINE': 'STATE-MACHINE',
        'Unspent Transaction Output': 'Unspent-Transaction-Output',
        'Virtual Machine': 'Virtual-Machine',
        'Zero-Knowledge Proof': 'Zero-Knowledge-Proof',
        'Zero Knowledge Proof': 'Zero-Knowledge-Proof',
        'NON-FUNGIBLE TOKEN': 'NON-FUNGIBLE-TOKEN',
        'NON FUNGIBLE TOKEN': 'NON-FUNGIBLE-TOKEN',
        'SECURITY TOKEN': 'SECURITY-TOKEN',
        'STABLE TOKEN': 'STABLE-TOKEN',
        'UTILITY TOKEN': 'UTILITY-TOKEN',
        'CANONICAL BLOCK': 'CANONICAL-BLOCK',
        'GENESIS BLOCK': 'GENESIS-BLOCK',
        'TOTAL COMPLETE': 'TOTAL-COMPLETE',
        'BLOCKCHAIN 1.0': 'BLOCKCHAIN-1.0',
        'TRANSACTION FEE': 'TRANSACTION-FEE',
        'TRANSACTION POOL': 'TRANSACTION-POOL',
        'BLOCKCHAIN 2.0': 'BLOCKCHAIN-2.0',
        'BLOCKCHAIN 3.0': 'BLOCKCHAIN-3.0',
        'TURING COMPLETE': 'TURING-COMPLETE',
        'TURING MACHINE': 'TURING-MACHINE',
        'DECENTRALIZED EXCHANGE': 'DECENTRALIZED-EXCHANGE',
        'HARD FORK': 'HARD-FORK',
        'SOFT FORK': 'SOFT-FORK',
        'OPEN SOURCE': 'OPEN-SOURCE',
        'PRIVATE KEY INFRASTRUCTURE': 'PRIVATE KEY INFRASTRUCTURE',
        'DELEGATED PROOF OF STAKE': 'DELEGATED-PROOF-OF-STAKE',
        'DELEGATED PROOF-OF-STAKE': 'DELEGATED-PROOF-OF-STAKE',
        'DELEGATED PROOF OF WORK': 'DELEGATED-PROOF-OF-WORK',
        'DELEGATED PROOF-OF-WORK': 'DELEGATED-PROOF-OF-WORK',
        'UNSPENT TRANSACTION OUTPUT': 'UNSPENT-TRANSACTION-OUTPUT',
        'VIRTUAL MACHINE': 'VIRTUAL-MACHINE',
        'MULTI SIGNATURE WALLET': 'MULTI-SIGNATURE-WALLET',
        'MULTI-SIGNATURE WALLET': 'MULTI-SIGNATURE-WALLET',
        'WEB ASSEMBLY': 'WEB-ASSEMBLY'
    }

    # Iterate over the terms and replace them in the text
    for term, replacement in terms_to_replace.items():
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(replacement, text)

    return text


def custom_filter(text_list, stopwords_path):
    # Load stopwords
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()

    # Filter out stopwords and special characters
    filtered_text_list = [word for word in text_list if
                          word.lower() not in stopwords and len(word) > 1 and (word.isalnum() or '-' in word)]

    return filtered_text_list


def preprocess_data(data_type, input_path, output_path_base, stopwords_path):
    _, file_extension = os.path.splitext(input_path)
    if file_extension.lower() == '.csv':
        df = pd.read_csv(input_path, encoding="utf8").fillna("")
    elif file_extension.lower() == '.xlsx':
        df = pd.read_excel(input_path).fillna("")
    else:
        raise ValueError("Invalid file extension. Use '.csv' or '.xlsx'.")

    #df = df.head(500)

    if data_type == 'paper':
        df.rename(columns={'cover_date': 'date'}, inplace=True)
        df['keywords'] = df['keywords'].str.replace("|", "")
        df['text'] = df['title'] + " " + df['abstract'] + " " + df['keywords']
    elif data_type == 'news':
        df.rename(columns={'pubdate': 'date'}, inplace=True)
        df['text'] = df['Title'] + " " + df['full text']
    elif data_type == 'patent':
        df['text'] = df['title'] + " " + df['abstract']
    else:
        raise ValueError("Invalid data type. Choose from 'paper', 'news', 'patent'.")

    # Preprocess text
    df['text'] = df['text'].apply(preprocess_text).str.lower()

    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Word(),
                            ptm.tagger.NLTK(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(1, 1))
    result = pipeline.processCorpus(df['text'].tolist())
    
    # df['text'] = result
    
    for i, text_list in enumerate(result):
        # Flatten list and apply custom filter
        flat_list = sum(text_list, [])
        # Remove duplicates without preserving order
        flat_list = list(set(flat_list))
        df.at[i, 'text'] = custom_filter(flat_list, stopwords_path)


    pickle_path = f"{output_path_base}.pkl"
    csv_path = f"{output_path_base}.csv"
    tsv_path = f"{output_path_base}.tsv"

    with open(pickle_path, "wb") as file:
        pickle.dump(df, file)

    df.to_csv(csv_path, index=False)
    df.to_csv(tsv_path, sep='\t', index=False)

    return df.head()


# Example usage
data_type = 'patent'
input_path = './data/patents_201701_202312_sample.xlsx'
output_path_base = './results/patents_201701_202312_sample'
stopwords_path = './stopwords/Stopword_Eng_Blockchain.txt'

# Call the function
preprocessed_head = preprocess_data(data_type, input_path, output_path_base, stopwords_path)
print(preprocessed_head)
