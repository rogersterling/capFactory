import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = 'https://pitch.vc'

def get_company_links():
    print("Starting to collect company links...")
    company_links = []
    page = 1
    while True:
        print(f"Scanning page {page}...")
        response = requests.get(f"{BASE_URL}/companies?page={page}")
        if response.status_code != 200:
            print(f"Failed to fetch page {page}: Status code {response.status_code}")
            break
        
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=lambda x: x and x.startswith('/companies/'))
        new_links = [link['href'] for link in links if 'company-logo' in str(link)]
        
        if not new_links:
            print("No more companies found.")
            break
        
        company_links.extend(new_links)
        print(f"Found {len(new_links)} companies on page {page} (Total: {len(company_links)})")
        
        page += 1
        time.sleep(2)
        
    unique_links = list(set(company_links))
    print(f"\nTotal unique companies found: {len(unique_links)}\n")
    return unique_links

def scrape_company_data(company_url):
    try:
        response = requests.get(company_url)
        if response.status_code != 200:
            print(f"Failed to fetch {company_url}: Status code {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Initialize data with default values
        data = {
            'company_name': '',
            'tagline': '',
            'website': '',
            'location': '',
            'description': '',
            'tags': [],
            'team': [],
            'problem_statement': '',
            'milestones': [],
            'founded_date': '',
            'last_updated': ''
        }
        
        # Get company name and tagline
        name_elem = soup.find('h1', class_='text-3xl')
        if not name_elem:
            print(f"Warning: Could not find company name for {company_url}")
            return None
        data['company_name'] = name_elem.text.strip()
        print(f"Scraping data for: {data['company_name']}")
        
        # Get tagline (now looking in the correct location)
        tagline_elem = soup.find('div', class_='text-xs text-gray-400 max-w-xl mt-2')
        if tagline_elem:
            data['tagline'] = tagline_elem.text.strip()
        
        # Get website, location, and dates
        social_section = soup.find('div', class_='bg-gray-200/30')
        if social_section:
            website_link = social_section.find('a', class_='social-link')
            if website_link:
                data['website'] = website_link.get('href', '')
            
            # Get location
            location_elem = social_section.find('a', href=lambda x: x and 'filter' in x and 'cities' in x)
            if location_elem:
                data['location'] = location_elem.text.strip()
            
            # Get founded date and last updated
            dates = social_section.find_all('span', class_='hidden md:inline')
            for date in dates:
                if 'Founded' in date.text:
                    data['founded_date'] = date.text.replace('Founded ', '')
                elif 'Profile updated' in date.text:
                    data['last_updated'] = date.text.replace('Profile updated ', '')
        
        # Get description and problem statement
        text_sections = soup.find_all('div', class_='text')
        if text_sections:
            if len(text_sections) > 0:
                data['description'] = text_sections[0].get_text(strip=True)
            for section in text_sections:
                if section.find_previous('h2', class_='section-headline') and 'Problem statement' in section.find_previous('h2', class_='section-headline').text:
                    data['problem_statement'] = section.get_text(strip=True)
        
        # Get tags
        tag_elements = soup.find_all('span', class_='tag')
        if tag_elements:
            data['tags'] = [tag.text.strip() for tag in tag_elements if tag.text]
        
        # Get team information (updated selector)
        team_grid = soup.find('div', class_='grid gap-4 md:grid-cols-2')
        if team_grid:
            for member in team_grid.find_all('div', class_='border rounded-lg shadow-sm bg-gray-50'):
                try:
                    name_div = member.find('div', class_='flex-grow text-base py-2 px-3')
                    if name_div:
                        name = name_div.contents[0].strip()
                        role = name_div.find('div', class_='text-sm text-gray-400').text.strip()
                        data['team'].append({
                            'name': name,
                            'role': role
                        })
                except Exception as e:
                    print(f"Error parsing team member in {data['company_name']}: {str(e)}")
                    continue
        
        # Get milestones
        milestones_section = soup.find_all('div', class_='border rounded-lg p-6')
        for milestone in milestones_section:
            try:
                date_div = milestone.find('div', class_='float-right')
                title = milestone.find('h2', class_='font-semibold text-base mb-4')
                description = milestone.find('div', class_='text-gray-500 text text-sm')
                
                if date_div and title:
                    data['milestones'].append({
                        'date': date_div.text.strip(),
                        'title': title.text.strip(),
                        'description': description.text.strip() if description else ''
                    })
            except Exception as e:
                print(f"Error parsing milestone in {data['company_name']}: {str(e)}")
                continue
        
        print(f"Successfully scraped data for {data['company_name']}")
        return data
        
    except Exception as e:
        print(f"Error scraping {company_url}: {str(e)}")
        return None

if __name__ == '__main__':
    print("Starting the scraping process...\n")
    
    company_links = get_company_links()
    all_data = []
    
    print("\nBeginning to scrape individual company pages...")
    for i, relative_link in enumerate(company_links, 1):
        company_url = f"{BASE_URL}{relative_link}"
        print(f"\nProcessing company {i} of {len(company_links)}")
        company_data = scrape_company_data(company_url)
        if company_data:
            all_data.append(company_data)
        time.sleep(1)
    
    print(f"\nScraping complete! Successfully scraped {len(all_data)} companies")
    print("Saving data to companies.json...")
    
    with open('companies.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print("Done! Data saved to companies.json") 