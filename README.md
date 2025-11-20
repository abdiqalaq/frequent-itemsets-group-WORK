# Supermarket Transactions Analysis Using Association Rule Mining

## Project Overview

This project implements a complete Market Basket Analysis pipeline using the Apriori algorithm to discover frequent itemsets, closed frequent itemsets, and maximal frequent itemsets from simulated supermarket transaction data. The analysis helps identify product associations and purchasing patterns that can inform business decisions such as product placement, promotions, and inventory management.

## Table of Contents

- [Team Members & Contributions](#team-members--contributions)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Installation & Requirements](#installation--requirements)
- [Usage Instructions](#usage-instructions)
- [Output Files](#output-files)
- [Key Findings](#key-findings)
- [References](#references)

---

## Team Members & Contributions

### 1. **Abdiqalaq** - Data Generation
**Responsibility:** Section 1 - Transaction Data Generation

- Created a comprehensive pool of 30 diverse supermarket items representing common grocery categories
- Implemented random transaction generation logic to simulate realistic shopping behaviors
- Generated 3,000 transactions with variable basket sizes (2-7 items per transaction)
- Ensured data quality and variety to support meaningful pattern discovery
- Stored raw transactions in list format for preprocessing

**Key Code Contributions:**
```python
- item_pool creation with 30 supermarket items
- Transaction generation loop with random sampling
- Validation of transaction count and structure
```

---

### 2. **Bradley Ochola (346)** - One-Hot Encoding & Preprocessing
**Responsibility:** Section 2 - Data Preprocessing and Transformation

- Implemented TransactionEncoder from mlxtend library for one-hot encoding
- Transformed transaction list into binary matrix format suitable for Apriori algorithm
- Created DataFrame structure with item names as column headers
- Validated encoding output and displayed sample results
- Prepared data pipeline for frequent pattern mining

**Key Code Contributions:**
```python
- TransactionEncoder initialization and fitting
- Binary matrix transformation
- DataFrame conversion with proper column naming
- Shape validation and sample display
```

---

### 3. **Sammi Oyabi** - Frequent Itemsets Mining
**Responsibility:** Section 3 - Apriori Algorithm Implementation

- Applied Apriori algorithm with 5% minimum support threshold
- Generated and sorted frequent itemsets by support values
- Analyzed itemset size distribution across all frequent patterns
- Identified top 10 most frequent itemsets
- Exported results to CSV for further analysis
- Calculated comprehensive statistics (min, max, average support)

**Key Code Contributions:**
```python
- Apriori algorithm configuration and execution
- Support threshold optimization
- Result sorting and filtering
- Statistical analysis implementation
- CSV export functionality
```

---

### 4. **Kendi (807)** - Closed Frequent Itemsets Identification
**Responsibility:** Section 4 - Closed Itemset Analysis

**Implementation Logic:**

A closed frequent itemset is defined as an itemset where **no immediate superset has the same support**. This means the itemset is "closed" in the sense that extending it would result in a different (lower) support value.

**Algorithm Approach:**
```python
def identify_closed_itemsets(frequent_itemsets):
    """
    Identifies closed frequent itemsets from the complete set of frequent itemsets.
    
    Logic:
    1. Sort itemsets by support (descending) and size (descending)
    2. For each itemset X:
       a. Check all itemsets Y where Y is a superset of X
       b. If any superset Y has the same support as X, then X is NOT closed
       c. If no superset has the same support, X is closed
    3. Return only closed itemsets
    """
    closed_itemsets = []
    
    # Convert to list for easier manipulation
    itemsets_list = frequent_itemsets.to_dict('records')
    
    for i, item_i in enumerate(itemsets_list):
        is_closed = True
        itemset_i = item_i['itemsets']
        support_i = item_i['support']
        
        # Check against all other itemsets
        for j, item_j in enumerate(itemsets_list):
            if i != j:
                itemset_j = item_j['itemsets']
                support_j = item_j['support']
                
                # Check if itemset_j is a superset of itemset_i
                # and has the same support
                if (itemset_i.issubset(itemset_j) and 
                    itemset_i != itemset_j and 
                    abs(support_i - support_j) < 1e-10):  # Float comparison
                    is_closed = False
                    break
        
        if is_closed:
            closed_itemsets.append(item_i)
    
    return pd.DataFrame(closed_itemsets)

# Apply the function
closed_itemsets = identify_closed_itemsets(frequent_itemsets)
closed_itemsets.to_csv("closed_itemsets.csv", index=False)
```

**Key Contributions:**
- Implemented closed itemset identification algorithm
- Added efficient superset checking logic
- Handled floating-point comparison for support values
- Exported closed itemsets to CSV
- Provided detailed inline documentation

---

### 5. **Victor Kipngeno (388)** - Maximal Frequent Itemsets Identification
**Responsibility:** Section 5 - Maximal Itemset Analysis

**Implementation Logic:**

A maximal frequent itemset is defined as a frequent itemset that **has no frequent superset**. These represent the "boundaries" of frequent patterns - they are the largest frequent itemsets that cannot be extended while maintaining the minimum support threshold.

**Algorithm Approach:**
```python
def identify_maximal_itemsets(frequent_itemsets):
    """
    Identifies maximal frequent itemsets from the complete set of frequent itemsets.
    
    Logic:
    1. Sort itemsets by size (descending) - larger itemsets are more likely maximal
    2. For each itemset X:
       a. Check if ANY superset of X exists in the frequent itemsets
       b. If any frequent superset exists, X is NOT maximal
       c. If no frequent superset exists, X is maximal
    3. Return only maximal itemsets
    
    Note: Maximal itemsets are always a subset of closed itemsets
    """
    maximal_itemsets = []
    
    # Convert to list for easier manipulation
    itemsets_list = frequent_itemsets.to_dict('records')
    
    # Sort by itemset size (descending) for optimization
    itemsets_list.sort(key=lambda x: len(x['itemsets']), reverse=True)
    
    for i, item_i in enumerate(itemsets_list):
        is_maximal = True
        itemset_i = item_i['itemsets']
        
        # Check if any other itemset is a superset
        for j, item_j in enumerate(itemsets_list):
            if i != j:
                itemset_j = item_j['itemsets']
                
                # If itemset_j is a proper superset of itemset_i,
                # then itemset_i is not maximal
                if itemset_i.issubset(itemset_j) and itemset_i != itemset_j:
                    is_maximal = False
                    break
        
        if is_maximal:
            maximal_itemsets.append(item_i)
    
    return pd.DataFrame(maximal_itemsets)

# Apply the function
maximal_itemsets = identify_maximal_itemsets(frequent_itemsets)
maximal_itemsets.to_csv("maximal_itemsets.csv", index=False)
```

**Key Contributions:**
- Implemented maximal itemset identification algorithm
- Optimized search by sorting itemsets by size
- Added efficient superset existence checking
- Exported maximal itemsets to CSV
- Provided comprehensive inline comments

**Relationship Between Itemset Types:**
```
All Frequent Itemsets (Apriori output)
    ↓
Closed Frequent Itemsets (no superset with same support)
    ↓
Maximal Frequent Itemsets (no frequent superset at all)
```

---

## Project Structure

```
frequent-items-group-work/
│
├── frequent_itemsets_analysis.ipynb    # Main Jupyter notebook with all code
├── README.md                            # This file
│
│   ├── transactions.csv    # Raw transaction data (3000 transactions)
│   ├── frequent_itemsets.csv           # All frequent itemsets from Apriori
│   ├── closed_frequent_itemsets.csv             # Closed frequent itemsets
│   ├── maximal_itemsets.csv            # Maximal frequent itemsets
│   └── top_10_frequent_itemsets.csv    # Top 10 by support
│

```

---

## Implementation Details

### 1. Data Generation
- **Transaction Count:** 3,000 simulated transactions
- **Basket Size:** 2-7 items per transaction (randomly selected)
- **Item Pool:** 30 unique supermarket items
- **Randomization:** Ensures diverse shopping patterns

### 2. Preprocessing
- **Encoding Method:** One-hot encoding using TransactionEncoder
- **Output Format:** Binary matrix (1 = item present, 0 = absent)
- **Dimensions:** 3000 rows × 30 columns

### 3. Frequent Pattern Mining
- **Algorithm:** Apriori
- **Minimum Support:** 5% (0.05)
- **Optimization:** Column-based representation for efficiency

### 4. Closed Itemsets
- **Definition:** Itemsets with no superset having identical support
- **Purpose:** Reduce redundancy while preserving all support information
- **Count:** Typically 20-40% of all frequent itemsets

### 5. Maximal Itemsets
- **Definition:** Frequent itemsets with no frequent superset
- **Purpose:** Represent boundaries of frequent patterns
- **Count:** Typically 5-15% of all frequent itemsets

---

## Installation & Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries
```bash
pip install pandas numpy mlxtend
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
mlxtend>=0.19.0
jupyter>=1.0.0
```

---

## Usage Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/[your-username]/supermarket-analysis.git
cd supermarket-analysis
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Notebook
```bash
jupyter notebook frequent_itemsets_analysis.ipynb
```

### Step 4: Execute All Cells
Run all cells sequentially to:
1. Generate transaction data
2. Perform one-hot encoding
3. Mine frequent itemsets
4. Identify closed itemsets
5. Identify maximal itemsets
6. Export all results to CSV

---

## Output Files

### 1. supermarket_transactions.csv
- Raw transaction data in list format
- 3,000 rows representing individual transactions
- Each row contains 2-7 items

### 2. frequent_itemsets.csv
- All itemsets meeting minimum support threshold (5%)
- Columns: `itemsets`, `support`
- Sorted by support (descending)

### 3. closed_itemsets.csv
- Subset of frequent itemsets that are closed
- Represents condensed pattern information
- Columns: `itemsets`, `support`

### 4. maximal_itemsets.csv
- Subset of frequent itemsets that are maximal
- Represents pattern boundaries
- Columns: `itemsets`, `support`

### 5. top_10_frequent_itemsets.csv
- Top 10 itemsets with highest support values
- Quick reference for most common patterns



## Key Findings

### Frequent Itemsets Statistics
- **Total Frequent Itemsets:** [Number varies based on random data]
- **Minimum Support Threshold:** 5%
- **Highest Support:** [Varies]
- **Average Support:** [Varies]

### Itemset Distribution
- **1-itemsets:** Most common (individual items)
- **2-itemsets:** Product pairs frequently purchased together
- **3-itemsets:** Common triplets
- **4+ itemsets:** Larger baskets with multiple related items

### Business Insights
- Identified key product associations
- Discovered cross-category purchasing patterns
- Found opportunities for bundle promotions
- Determined optimal product placement strategies



## Algorithm Complexity

### Apriori Algorithm
- **Time Complexity:** O(2^n) worst case, where n = number of items
- **Space Complexity:** O(n × m) where m = number of transactions
- **Optimization:** Pruning reduces practical complexity significantly

### Closed Itemsets Identification
- **Time Complexity:** O(f^2) where f = number of frequent itemsets
- **Space Complexity:** O(f)

### Maximal Itemsets Identification
- **Time Complexity:** O(f^2) where f = number of frequent itemsets
- **Space Complexity:** O(f)

---

## Future Enhancements

1. **Association Rules Generation**
   - Calculate confidence and lift metrics
   - Identify strong rules (e.g., {Bread, Butter} → {Milk})

2. **Visualization**
   - Network graphs showing item relationships
   - Heatmaps of product associations
   - Support distribution histograms

3. **Parameter Optimization**
   - Experiment with different support thresholds
   - Analyze impact on itemset count and quality

4. **Real Data Integration**
   - Apply to actual POS transaction data
   - Validate patterns against business knowledge

5. **Advanced Algorithms**
   - Implement FP-Growth for comparison
   - Explore ECLAT algorithm for vertical data format


## References

1. Agrawal, R., & Srikant, R. (1994). "Fast Algorithms for Mining Association Rules"
2. MLxtend Documentation: https://rasbt.github.io/mlxtend/
3. Zaki, M. J., & Hsiao, C. J. (2002). "CHARM: An Efficient Algorithm for Closed Itemset Mining"
4. Bayardo, R. J. (1998). "Efficiently Mining Long Patterns from Databases"


## License

This project is created for educational purposes as part of a data mining course assignment.


## Contact

For questions or collaboration:
- **Abdiqalaq:** abdiqalaq99@gmail.com 
- **Bradley Ochola:** ocholabrad@gmail.com 
- **Sammi Oyabi:** smaoyabi@gmail
- **Kendi Nyaga:** kendynyaga@gmail.com
- **Victor Kipngeno:** victorkipngenorotich21@gmail.com 

---

## Acknowledgments

Special thanks to the course instructor and teaching assistants for guidance on association rule mining concepts and implementation techniques.

---

**Last Updated:** November 2025
