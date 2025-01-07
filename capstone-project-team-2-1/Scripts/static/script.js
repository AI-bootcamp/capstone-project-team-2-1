const emojis = {
    'Apps': '📱', 'Commercial': '💼', 'Conferences and Events': '🎪',
    'Education': '🎓', 'Financial': '💰', 'Governmental': '🏛️',
    'Health': '🏥', 'Personal': '👤', 'Promotional': '📢',
    'Services': '🛠️', 'Stores': '🏪', 'Telecommunications': '📞',
    'Travel': '✈️'
};

let currentLanguage = 'ar'; // Default to Arabic
let isDarkMode = false;

// Load categories and display cards
async function loadCategories() {
    const response = await fetch('/get_categories');
    const data = await response.json();
    const cardsContainer = document.getElementById('category-cards');
    cardsContainer.innerHTML = '';
    data.categories.forEach(category => {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
            ${category.emoji} 
            <span class="category-name">${currentLanguage === 'ar' ? category.ar : category.en}</span>
            <div class="count">${category.count} ${currentLanguage === 'ar' ? 'رسائل' : 'messages'}</div>
        `;
        card.onclick = () => openSidebar(category.en);
        cardsContainer.appendChild(card);
    });
}

// Open sidebar and load messages
async function openSidebar(category) {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.add('open');
    const response = await fetch(`/get_messages/${category}`);
    const data = await response.json();
    const messagesContainer = document.getElementById('messages-container');
    messagesContainer.innerHTML = '';
    data.messages.forEach(message => {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message';
        messageDiv.innerHTML = `
            <h3>${message.Sender}</h3>
            <p>${message['Message Content']}</p>
        `;
        messagesContainer.appendChild(messageDiv);
    });
}

// Close sidebar
function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.remove('open');
}

// Toggle language
function toggleLanguage() {
    currentLanguage = currentLanguage === 'ar' ? 'en' : 'ar';
    document.getElementById('title').textContent = currentLanguage === 'ar' ? 'الرسائل' : 'Messages';
    document.getElementById('language-icon').textContent = currentLanguage === 'ar' ? '🇸🇦' : '🇺🇸'; // Use different emojis
    loadCategories();
}

// Toggle theme
function toggleTheme() {
    isDarkMode = !isDarkMode;
    document.body.classList.toggle('dark-mode', isDarkMode);
    document.getElementById('theme-icon').textContent = isDarkMode ? '🌞' : '🌙';
}

// Initialize
loadCategories();