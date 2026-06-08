import { Marp } from '@marp-team/marp-core'
import markdownItContainer from 'markdown-it-container'

const defineContainer = (name, cssClass) => [
  markdownItContainer,
  name,
  {
    render(tokens, idx) {
      return tokens[idx].nesting === 1
        ? `<div class="${cssClass}">\n`
        : `</div>\n`
    },
  },
]

export default class Engine extends Marp {
  constructor(opts) {
    super(opts)
    this.markdown
      .use(...defineContainer('columns',       'columns'))
      .use(...defineContainer('columns-60-40', 'columns columns--60-40'))
      .use(...defineContainer('columns-40-60', 'columns columns--40-60'))
      .use(...defineContainer('col',           'col'))
  }
}
