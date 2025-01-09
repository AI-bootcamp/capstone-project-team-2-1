document.addEventListener('DOMContentLoaded', function () {
  // Dark mode toggle with localStorage
  const themeToggle = document.getElementById('theme-toggle');
  const body = document.body;

  // Check saved theme
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) {
    body.dataset.theme = savedTheme;
    updateThemeIcon(savedTheme); // Update icon based on saved theme
  }

  // Toggle theme
  themeToggle.addEventListener('click', function () {
    const newTheme = body.dataset.theme === 'dark' ? 'light' : 'dark';
    body.dataset.theme = newTheme;
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme); // Update icon when theme changes
  });

  // Update theme icon
  function updateThemeIcon(theme) {
    const icon = theme === 'dark' ? '🌞' : '🌙'; // Sun icon for dark mode, moon icon for light mode
    themeToggle.innerHTML = icon;
  }

  // Filter messages by importance with animations
  const importanceFilter = document.getElementById('importance-filter');
  if (importanceFilter) {
    importanceFilter.addEventListener('change', function () {
      const showImportantOnly = this.value === 'important';
      document.querySelectorAll('.card').forEach(card => {
        const messages = card.querySelectorAll('.message');
        let importantCount = 0;

        messages.forEach(message => {
          if (showImportantOnly && message.classList.contains('not-important')) {
            message.style.opacity = '0';
            setTimeout(() => {
              message.style.display = 'none';
            }, 300); // Match the transition duration
          } else {
            message.style.display = 'block';
            setTimeout(() => {
              message.style.opacity = '1';
            }, 10);
            if (!message.classList.contains('not-important')) {
              importantCount++;
            }
          }
        });

        // Update the card header count with animation
        const cardHeader = card.querySelector('h3');
        const category = cardHeader.textContent.split(' (')[0];
        const countText = showImportantOnly
          ? `${importantCount} رسالة مهمة`
          : `${messages.length} رسالة`;

        // Add a small counter under the title
        const counter = card.querySelector('.message-counter') || document.createElement('div');
        counter.className = 'message-counter';
        counter.textContent = countText;
        cardHeader.insertAdjacentElement('afterend', counter);

        // Animate the counter
        counter.style.opacity = '0';
        setTimeout(() => {
          counter.style.opacity = '1';
        }, 300);
      });
    });
  }

  // Modal functionality
  const modal = document.getElementById('message-modal');
  const closeModalBtn = document.querySelector('.close-modal');

  // Open modal with messages based on filter
  document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('click', function () {
      const category = this.querySelector('h3').textContent.split(' (')[0]; // Extract category name
      const showImportantOnly = importanceFilter ? importanceFilter.value === 'important' : false;
      const messages = Array.from(this.querySelectorAll('.message'))
        .filter(message => showImportantOnly ? !message.classList.contains('not-important') : true)
        .map(message => {
          // Preserve the class for styling
          const isImportant = message.classList.contains('important');
          return `
            <div class="message ${isImportant ? 'important' : 'not-important'}">
              <p><strong>المرسل:</strong> ${message.querySelector('p:nth-child(1)').textContent}</p>
              <p><strong>المحتوى:</strong> ${message.querySelector('p:nth-child(2)').textContent}</p>
            </div>
          `;
        });

      // Update modal content
      document.getElementById('modal-category').textContent = category;
      document.getElementById('modal-messages').innerHTML = messages.join('<hr>');
      modal.style.display = 'flex'; // Show the modal
    });
  });

  // Close modal with "X" button
  if (closeModalBtn) {
    closeModalBtn.addEventListener('click', function () {
      modal.style.display = 'none';
    });
  }

  // Close modal when clicking outside
  window.addEventListener('click', function (event) {
    if (event.target === modal) {
      modal.style.display = 'none';
    }
  });

  // Make modal draggable
  let isDragging = false;
  let offsetX, offsetY;

  const modalContent = document.querySelector('.modal-content');
  if (modalContent) {
    modalContent.addEventListener('mousedown', function (e) {
      isDragging = true;
      offsetX = e.clientX - modalContent.getBoundingClientRect().left;
      offsetY = e.clientY - modalContent.getBoundingClientRect().top;
    });

    window.addEventListener('mousemove', function (e) {
      if (isDragging) {
        modalContent.style.left = `${e.clientX - offsetX}px`;
        modalContent.style.top = `${e.clientY - offsetY}px`;
      }
    });

    window.addEventListener('mouseup', function () {
      isDragging = false;
    });
  }
});