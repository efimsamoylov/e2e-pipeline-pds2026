def rule_based_department(text: str) -> str:
    """
    Rule-based department detection.
    Returns department label or None.
    """
    t = text.lower()

    # 1. Specific Departments (Check first)
    
    # IT
    it_keywords = [
        'software', 'developer', 'engineer', 'programmer', 'architect', 'devops',
        'data scientist', 'data engineer', 'data analyst', 'machine learning', 'ai ',
        'system', 'network', 'database', 'infrastructure', 'cloud', 'sysadmin',
        'security', 'cybersecurity', 'infosec',
        'technical', 'technology', 'tech ', 'it ', 'qa engineer', 'test engineer',
        'frontend', 'backend', 'full stack', 'fullstack', 'web developer',
        'java', 'python', 'javascript', '.net', 'c++', 'ruby', 'php',
        'applikation', 'informatik', 'ingenieur', 'cto', 'cio'
    ]
    if any(kw in t for kw in it_keywords):
        return "Information Technology"

    # Sales
    sales_keywords = [
        'sales', 'verkauf', 'vertrieb', 'vendas', 'ventes', 'ventas',
        'account executive', 'account manager', 'business development',
        'commercial', 'verkaufsleiter', 'verkaufer', 'vertriebsleiter',
        'inside sales', 'outside sales', 'field sales', 'key account'
    ]
    if any(kw in t for kw in sales_keywords):
        # Exclusion for C-level (handled in Other) unless explicitly Sales C-level
        if not any(ek in t for ek in ['ceo', 'founder', 'owner']):
            return "Sales"

    # Marketing
    marketing_keywords = [
        'marketing', 'communication', 'kommunikation', 'comunicacao',
        'brand', 'content', 'social media', 'digital marketing', 'online marketing',
        'seo', 'sem', 'ppc', 'campaign', 'advertising', 'public relations', 'pr ',
        'media', 'creative', 'copywriter', 'graphic', 'webmarketing', 'growth'
    ]
    if any(kw in t for kw in marketing_keywords):
        return "Marketing"

    # 2. "Other" Category (Executives, Operations, etc.)
    
    executive_keywords = ['ceo', 'cfo', 'founder', 'owner', 'geschaftsfuhrer', 
                         'managing director', 'prokurist', 'vorstand', 'board member']
    if any(kw in t for kw in executive_keywords):
        return "Other"

    academic_keywords = ['professor', 'dr phil', 'physician', 'doctor', 'researcher', 'scientist']
    if any(kw in t for kw in academic_keywords):
        return "Other"

    operations_keywords = ['operations director', 'head of operations', 'operations manager',
                          'chief operations', 'operations officer']
    if any(kw in t for kw in operations_keywords):
        return "Other"

    finance_keywords = ['finance director', 'financial controller', 'treasurer', 
                       'kreditsachbearbeiter', 'betriebswirt']
    if any(kw in t for kw in finance_keywords):
        return "Other"

    return None


def rule_based_seniority(text: str) -> str:
    """
    Rule-based seniority detection.
    Returns seniority label or None.
    """
    t = text.lower()

    # Director
    if 'director' in t and 'associate' not in t:
        return "Director"
    
    # Head of / VP
    if any(kw in t for kw in ['head of', 'vice president', 'vp ', 'vp,']):
        return "Director"

    # C-Level / Management (High level)
    if any(kw in t for kw in ['ceo', 'cfo', 'cto', 'coo', 'founder', 'owner', 'chief', 
                              'general manager', 'managing director', 'geschaftsfuhrer']):
        return "Management"

    # Lead / Manager (Mid level)
    # Note: "Manager" is often Lead in this dataset context, not Executive Management
    if any(kw in t for kw in ['manager', 'managing', 'lead', 'principal', 'chef de', 'coordinator', 'specialist']):
        return "Lead"

    # Senior
    if 'senior' in t or 'sr ' in t or 'sr.' in t:
        return "Senior"

    # Junior
    if any(kw in t for kw in ['junior', 'jr ', 'jr.', 'associate', 'analyst', 'trainee', 'intern', 'student']):
        return "Junior"

    return None
