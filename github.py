import requests
import os
def get_recent_commits_old(token, per_page=30):
    url = f"https://api.github.com/repos/SynthstromAudible/DelugeFirmware/commits"
    headers = {"Authorization": f"token {token}"}
    params = {"per_page": per_page}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        commits = response.json()
        return commits
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def get_recent_commits(token, page=1, per_page=30, max_commits=100, commits = []):
    """
    Get recent commits from a GitHub repository, handling pagination.

    Parameters:
    - username: The GitHub username of the repository owner.
    - repo_name: The name of the GitHub repository.
    - page: The page number to retrieve (default is 1).
    - per_page: The number of commits per page (default is 30).

    Returns:
    A list of dictionaries, where each dictionary represents a commit.
    """
    # GitHub API endpoint for listing commits
    if page > 10:
        return commits
    if len(commits) >= max_commits:
        return commits

    url = f'https://api.github.com/repos/SynthstromAudible/DelugeFirmware/commits'
    headers = {"Authorization": f"token {token}"}
    # Parameters for the API request
    params = {
        'page': page,
        'per_page': per_page,
    }

    # Make the API request
    response = requests.get(url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
     
        # Parse the JSON response
        resp = response.json()
        commits += [c["sha"] for c in resp]
        # Check if there are more pages
        if 'Link' in response.headers:
            link_header = response.headers['Link']
            next_page_url = find_next_page_url(link_header)
            if next_page_url:
                # Recursively fetch more pages and append to the current list
                get_recent_commits(token, page + 1, per_page, max_commits, commits)
        return commits
    else:
        # Print an error message if the request was not successful
        print(f"Error: Unable to fetch commits (status code {response.status_code})")
        return None
    

def find_next_page_url(link_header):
    """
    Helper function to extract the URL for the next page from the 'Link' header.

    Parameters:
    - link_header: The 'Link' header from the GitHub API response.

    Returns:
    The URL for the next page, or None if no next page is found.
    """
    links = link_header.split(',')
    for link in links:
        parts = link.split(';')
        if 'rel="next"' in parts[1]:
            return parts[0].strip('<>')
    return None


if __name__ == "__main__":
    token = os.environ["GITHUB_TOKEN"]
    recent_commits = get_recent_commits(token, per_page=30, max_commits=70)
    if recent_commits:
        for commit in recent_commits:
            print(commit)
